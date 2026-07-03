"""预算优化的不确定性量化：block bootstrap 置信区间。

为什么需要它
------------
``budget_optimizer.optimize_budget`` 输出每个渠道的「最优 spend」是一个**单点
估计**——CMO 拿这个数字去砍预算是危险的：它隐含「我精确地知道最优分配」的错觉，
但事实上 Ridge 系数本身有噪声、训练样本只是历史的一小段。

本模块用 **block bootstrap** 给每个渠道的 `optimal_spend` 与总 revenue 提升配
**95% 置信区间**，把「会优化」升级为「懂决策风险」。

为什么是 *block* bootstrap（方法学关键）
-----------------------------------------
朴素 case-resample（逐行有放回重抽样）假设样本 i.i.d.。但 MMM 残差强自相关——
本项目 OLS 的 Durbin-Watson = 0.90（远小于无自相关应有的 2.0）。在强自相关下
逐行重抽样等于「假装样本独立」，会**系统性低估**不确定性、CI 偏窄、给 CMO 错
误的信心。

block bootstrap：把时间序列切成连续的 block（默认 7 天一块），**有放回重抽样
整块、块内保持原始时间顺序**地拼成新训练集——保留块内自相关结构。对比测试
（``tests/test_uncertainty.py::test_block_ci_wider_than_naive``）实证：在自相关
数据上 block bootstrap 的 CI 显著宽于朴素法，这正是诚实不确定性应有的样子。

管线
----
1. 加载 ``mmm_cleaned.parquet``，过滤出与 ``mmm_results.json`` **同一个 brand**
   （保持结果可比性），按 ``date_day`` 排序。
2. block bootstrap（N 次，默认 200）：block-resample → 复用
   ``mmm_model.fit_ridge`` 重拟合 → 复用 ``budget_optimizer.optimize_budget``
   重跑优化 → 记录每个渠道的 ``optimal_spend`` 与 ``optimal_revenue``。
3. 输出：每渠道 95% CI（百分位法）、revenue lift CI、点估计 vs 区间对比。
4. 产物：``reports/budget_uncertainty.json`` + ``reports/images/budget_ci.png``
   （森林图：每渠道一行，点 + 误差棒）。

复用而非复制：``mmm_model.prepare_features`` / ``chronological_split`` /
``fit_ridge``，以及 ``budget_optimizer.optimize_budget`` 全部直接 import。
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import (  # noqa: E402
    BLOCK_SIZE_DAYS,
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_RANDOM_SEED,
    CLEANED_PARQUET_PATH,
    IMAGES_DIR,
    MODEL_OUTPUT_DIR,
    N_BOOTSTRAP,
    REPORTS_DIR,
    SPEND_CHANNELS,
)
from scripts.budget_optimizer import (  # noqa: E402
    extract_params,
    load_mmm_results,
    optimize_budget,
)
from scripts.mmm_model import (  # noqa: E402
    chronological_split,
    fit_ridge,
    prepare_features,
)

# 朴素 case-resample 的对比块数（仅用于 block vs naive 宽度对比，保证可比）
_NAIVE_LABEL = "naive_case_resample"
_BLOCK_LABEL = "block_bootstrap"


# ---------------------------------------------------------------------------
# Block bootstrap 的核心：切块 + 重抽样
# ---------------------------------------------------------------------------


def make_blocks(n_rows: int, block_size: int) -> np.ndarray:
    """把 ``n_rows`` 行的（按时间排好序的）序列切成不重叠的连续 block。

    返回 ``(n_blocks, 2)`` 的 int 数组，每行是 ``[start, end)`` 的行索引区间。
    最后一块可能短于 ``block_size``（不足一块则合并进上一块，避免空块）。

    性质（在 tests/test_uncertainty.py 中断言）：
    - 块不重叠、覆盖全序列；
    - 块内行索引严格递增（保留时序）。
    """
    if block_size < 1:
        raise ValueError(f"block_size 必须 >= 1，得到 {block_size}")
    if n_rows < 1:
        raise ValueError(f"n_rows 必须 >= 1，得到 {n_rows}")

    starts = np.arange(0, n_rows, block_size)
    ends = np.clip(starts + block_size, None, n_rows)
    blocks = np.stack([starts, ends], axis=1)
    # 极端情况：最后一块若短到为空（理论上 clip 已避免），合并进上一块
    if len(blocks) > 1 and blocks[-1, 0] >= blocks[-1, 1]:
        blocks[-2, 1] = n_rows
        blocks = blocks[:-1]
    return blocks


def block_resample_indices(n_rows: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """block bootstrap 重抽样：有放回抽取整块，块内保持原始顺序拼接。

    标准 stationary/moving-block bootstrap  refill 策略：连续有放回抽取整块，
    直到累计长度达到 ``n_rows``，然后截断到恰好 ``n_rows``。返回长度恒等于
    ``n_rows`` 的索引数组，使重抽样后的训练集与原训练集等长（拟合规模一致，
    便于跨 bootstrap 次比较）。块内顺序被完整保留（块间才随机拼接）。
    """
    blocks = make_blocks(n_rows, block_size)

    out = np.empty(n_rows, dtype=np.int64)
    pos = 0
    while pos < n_rows:
        b_idx = int(rng.integers(0, len(blocks)))
        s, e = blocks[b_idx]
        seg = np.arange(s, e)
        take = min(len(seg), n_rows - pos)
        out[pos : pos + take] = seg[:take]
        pos += take
    return out


def case_resample_indices(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """朴素 case-resample：逐行有放回重抽样（**打乱时序**，破坏自相关）。

    保留它仅用于 block-vs-naive 的 CI 宽度对比，论证 block 的必要性。
    """
    return rng.integers(0, n_rows, size=n_rows)


# ---------------------------------------------------------------------------
# 单次 bootstrap 迭代：重拟合 Ridge → 重跑优化
# ---------------------------------------------------------------------------


def _one_bootstrap_iteration(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    feature_names: list[str],
    current_spend: dict[str, float],
    total_budget: float,
) -> dict | None:
    """一次 bootstrap 迭代：用重抽样索引重拟合 Ridge，再重跑预算优化。

    返回 ``{"optimal_spend": {...}, "optimal_revenue": float,
    "current_revenue": float, "elasticities": {...}, "converged": bool}``，
    或在拟合/优化失败时返回 ``None``（调用方跳过）。

    复用 ``mmm_model.fit_ridge`` 与 ``budget_optimizer.optimize_budget``。
    重拟合时仍用 chronological_split：重抽样后的序列仍是「时间顺序」的
    （block 内顺序被保留），保留尾部 holdout 评估的逻辑一致性。
    """
    X_b, y_b = X[idx], y[idx]
    n = len(y_b)
    # 用尾部 20% 作 holdout（与主管线一致的 frac），复用 chronological_split。
    dates_proxy = np.arange(n)  # fit_ridge 不需要真实日期，只要按序切分
    X_tr, X_te, y_tr, y_te = chronological_split(X_b, y_b, dates_proxy)

    try:
        ridge = fit_ridge(X_tr, y_tr, X_te, y_te, feature_names)
    except (ValueError, np.linalg.LinAlgError):
        return None

    # 把 Ridge 系数包装成 optimize_budget / extract_params 期望的格式
    mmm_like = {"models": {"ridge": ridge}}
    elasticities, intercept = extract_params(mmm_like)

    # 只保留有正 spend 的渠道（与主优化器口径一致，避免无意义渠道）
    active = {c: v for c, v in current_spend.items() if v > 0 and c in elasticities}
    if not active:
        return None

    try:
        opt = optimize_budget(active, elasticities, intercept, total_budget=total_budget)
    except (ValueError, RuntimeError):
        return None

    return {
        "optimal_spend": opt["optimal_spend"],
        "current_revenue": opt["current_revenue"],
        "optimal_revenue": opt["optimal_revenue"],
        "elasticities": elasticities,
        "converged": opt.get("converged", True),
    }


# ---------------------------------------------------------------------------
# 主 bootstrap 循环 + CI 汇总
# ---------------------------------------------------------------------------


def run_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    current_spend: dict[str, float],
    total_budget: float,
    n_bootstrap: int = N_BOOTSTRAP,
    block_size: int = BLOCK_SIZE_DAYS,
    seed: int = BOOTSTRAP_RANDOM_SEED,
    resample_fn=block_resample_indices,
    verbose: bool = True,
) -> dict:
    """运行 N 次 bootstrap，收集每渠道 optimal_spend 与 revenue 的分布。

    ``resample_fn`` 可换成 ``case_resample_indices`` 做 naive 对比（默认 block）。
    返回原始采样矩阵 ``{"optimal_spend": {channel: np.ndarray},
    "revenue_lift_pct": np.ndarray, "n_ok": int, "n_total": int,
    "resample": str}``，CI 由调用方 ``summarize_ci`` 计算。
    """
    rng = np.random.default_rng(seed)
    n_rows = len(y)

    spend_keys = [c for c, v in current_spend.items() if v > 0]
    spend_samples: dict[str, list[float]] = {c: [] for c in spend_keys}
    lift_samples: list[float] = []
    n_ok = 0

    label = "block" if resample_fn is block_resample_indices else "naive"
    for i in range(n_bootstrap):
        if resample_fn is block_resample_indices:
            idx = block_resample_indices(n_rows, block_size, rng)
        else:
            idx = case_resample_indices(n_rows, rng)

        res = _one_bootstrap_iteration(X, y, idx, feature_names, current_spend, total_budget)
        if res is None:
            continue
        n_ok += 1
        for c in spend_keys:
            spend_samples[c].append(float(res["optimal_spend"].get(c, np.nan)))
        cur, opt = res["current_revenue"], res["optimal_revenue"]
        lift = (opt - cur) / abs(cur) * 100.0 if cur not in (0, np.nan) else np.nan
        lift_samples.append(lift)

        if verbose and (i + 1) % 25 == 0:
            print(f"    [{label}] bootstrap {i + 1}/{n_bootstrap} 完成（成功 {n_ok}）")

    return {
        "optimal_spend": {c: np.array(v, dtype=float) for c, v in spend_samples.items()},
        "revenue_lift_pct": np.array(lift_samples, dtype=float),
        "n_ok": n_ok,
        "n_total": n_bootstrap,
        "resample": label,
    }


def summarize_ci(samples: dict, ci_level: float = BOOTSTRAP_CI_LEVEL) -> dict:
    """把 bootstrap 采样矩阵汇总成百分位 CI。

    对每个渠道的 ``optimal_spend``、以及 ``revenue_lift_pct`` 计算
    ``[alpha/2, 1-alpha/2]`` 百分位区间，同时给出中位数与均值。
    """
    alpha = 1.0 - ci_level
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)

    out = {
        "ci_level": ci_level,
        "n_successful": int(samples["n_ok"]),
        "n_total": int(samples["n_total"]),
        "resample": samples["resample"],
        "channels": {},
        "revenue_lift_pct": {},
    }

    for ch, arr in samples["optimal_spend"].items():
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        out["channels"][ch] = {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, lo_p)),
            "ci_high": float(np.percentile(arr, hi_p)),
            "ci_width": float(np.percentile(arr, hi_p) - np.percentile(arr, lo_p)),
            "n": int(len(arr)),
        }

    lift = samples["revenue_lift_pct"]
    lift = lift[np.isfinite(lift)]
    if len(lift) > 0:
        out["revenue_lift_pct"] = {
            "median": float(np.median(lift)),
            "mean": float(np.mean(lift)),
            "ci_low": float(np.percentile(lift, lo_p)),
            "ci_high": float(np.percentile(lift, hi_p)),
            "ci_width": float(np.percentile(lift, hi_p) - np.percentile(lift, lo_p)),
            "n": int(len(lift)),
        }
    return out


# ---------------------------------------------------------------------------
# 森林图（每渠道一行：点估计 + 误差棒）
# ---------------------------------------------------------------------------


def plot_forest(
    block_summary: dict,
    point_estimate: dict[str, float],
    output_path: Path,
    naive_summary: dict | None = None,
) -> None:
    """画森林图：每个渠道一行，block bootstrap 的 95% CI（点+误差棒）。

    叠加单点估计（红叉）以便对比「确定性优化器」给出的数字落在区间何处。
    若提供 ``naive_summary``，则同时画朴素 case-resample 的 CI（虚线）做对比——
    直观展示 naive CI 系统性偏窄（自相关被破坏的后果）。
    """
    channels = [c for c in point_estimate if c in block_summary["channels"]]
    if not channels:
        print("  无可绘制的渠道，跳过森林图。")
        return

    # 短标签，便于在 y 轴显示
    def short(c: str) -> str:
        return (
            c.replace("_spend", "")
            .replace("google_", "G·")
            .replace("meta_", "M·")
            .replace("tiktok", "TT")
        )

    labels = [short(c) for c in channels]
    y_pos = np.arange(len(channels))

    fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(channels) + 1.5)))

    block_lo = [block_summary["channels"][c]["ci_low"] for c in channels]
    block_hi = [block_summary["channels"][c]["ci_high"] for c in channels]
    block_med = [block_summary["channels"][c]["median"] for c in channels]
    point = [point_estimate[c] for c in channels]

    # 误差棒（block bootstrap CI）
    xerr_low = [m - lo for m, lo in zip(block_med, block_lo)]
    xerr_high = [hi - m for hi, m in zip(block_hi, block_med)]
    ax.errorbar(
        block_med,
        y_pos,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=2,
        capsize=5,
        label="Block bootstrap 95% CI",
        zorder=3,
    )

    # 单点估计（确定性优化器）
    ax.scatter(
        point,
        y_pos,
        marker="x",
        color="red",
        s=70,
        linewidths=2,
        label="Point estimate (deterministic optimizer)",
        zorder=4,
    )

    # 朴素 CI 对比（虚线）
    if naive_summary is not None:
        n_lo, n_hi, n_med = [], [], []
        for c in channels:
            d = naive_summary["channels"].get(c)
            if d:
                n_lo.append(d["ci_low"])
                n_hi.append(d["ci_high"])
                n_med.append(d["median"])
            else:
                n_lo.append(np.nan)
                n_hi.append(np.nan)
                n_med.append(np.nan)
        ax.errorbar(
            n_med,
            y_pos + 0.18,
            xerr=[
                [m - lo for m, lo in zip(n_med, n_lo)],
                [hi - m for hi, m in zip(n_hi, n_med)],
            ],
            fmt="none",
            color="#ff7f0e",
            ecolor="#ff7f0e",
            elinewidth=1.2,
            capsize=3,
            linestyle="--",
            label="Naive case-resample 95% CI",
            zorder=2,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Optimal daily spend ($)")
    ax.set_title(
        f"Budget Allocation Uncertainty — Block Bootstrap 95% CI\n"
        f"(block_size={BLOCK_SIZE_DAYS}d, N={block_summary['n_successful']} successful)"
    )
    ax.axvline(0, color="gray", linewidth=0.5, zorder=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"  森林图已保存到 {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口：组装数据 → bootstrap → 写报告
# ---------------------------------------------------------------------------


def load_brand_frame(brand_id: str) -> pl.DataFrame:
    """加载 mmm_cleaned.parquet 并过滤到指定 brand，按 date_day 排序。"""
    df = pl.read_parquet(CLEANED_PARQUET_PATH)
    sub = df.filter(pl.col("organisation_id") == brand_id).sort("date_day")
    if sub.height == 0:
        raise ValueError(f"parquet 中找不到 brand={brand_id} 的数据")
    return sub


def compute_current_spend(df: pl.DataFrame) -> dict[str, float]:
    """该 brand 各渠道的日均 spend，作为优化器的 current_spend 基线。

    与 ``budget_optimizer.main`` 的口径保持一致（取均值）。
    """
    spend = {}
    for ch in SPEND_CHANNELS:
        if ch not in df.columns:
            spend[ch] = 0.0
            continue
        avg = float(df[ch].mean())
        spend[ch] = avg if avg and avg > 0 else 0.0
    return spend


def run(n_bootstrap: int = N_BOOTSTRAP, block_size: int = BLOCK_SIZE_DAYS) -> dict:
    """完整不确定性管线：加载 brand → bootstrap → 写 json + 森林图。"""
    print("=" * 70)
    print("预算优化不确定性量化 — block bootstrap 置信区间")
    print("=" * 70)

    mmm = load_mmm_results()
    brand_id = mmm["brand_id"]
    print(f"  目标 brand（与 mmm_results.json 一致）: {brand_id}")

    df = load_brand_frame(brand_id)
    print(f"  加载 brand 数据：{df.height} 行，{df['date_day'].min()} → {df['date_day'].max()}")

    X, y, feature_names, _dates = prepare_features(df)
    current_spend = compute_current_spend(df)
    total_budget = float(sum(current_spend.values()))

    active = {c: v for c, v in current_spend.items() if v > 0}
    print(f"  活跃渠道数: {len(active)}，总预算（日均）: ${total_budget:,.0f}")
    print(f"  block_size={block_size} 天，N_BOOTSTRAP={n_bootstrap}")

    # 确定性单点估计（复用主优化器，作为区间内的参考点）
    point_elasticities, point_intercept = extract_params(mmm)
    point_opt = optimize_budget(
        active, point_elasticities, point_intercept, total_budget=total_budget
    )
    point_estimate = dict(point_opt["optimal_spend"])

    # ---- block bootstrap（主结果）----
    print("\n  [1/2] Block bootstrap（保留自相关）运行中...")
    block_samples = run_bootstrap(
        X,
        y,
        feature_names,
        current_spend,
        total_budget,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        seed=BOOTSTRAP_RANDOM_SEED,
        resample_fn=block_resample_indices,
        verbose=True,
    )
    block_summary = summarize_ci(block_samples)

    # ---- naive case-resample（对比，证明 block 必要）----
    print("\n  [2/2] Naive case-resample（破坏自相关，对比用）运行中...")
    naive_samples = run_bootstrap(
        X,
        y,
        feature_names,
        current_spend,
        total_budget,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        seed=BOOTSTRAP_RANDOM_SEED,
        resample_fn=case_resample_indices,
        verbose=True,
    )
    naive_summary = summarize_ci(naive_samples)

    # ---- 汇总 + 写报告 ----
    report = {
        "brand_id": brand_id,
        "n_rows": int(df.height),
        "config": {
            "block_size_days": block_size,
            "n_bootstrap": n_bootstrap,
            "ci_level": BOOTSTRAP_CI_LEVEL,
            "seed": BOOTSTRAP_RANDOM_SEED,
        },
        "point_estimate": point_estimate,
        "point_revenue": {
            "current_revenue": point_opt["current_revenue"],
            "optimal_revenue": point_opt["optimal_revenue"],
            "improvement_pct": point_opt["improvement_pct"],
        },
        "block_bootstrap_ci": block_summary,
        "naive_case_resample_ci": naive_summary,
        "block_vs_naive_width": _width_comparison(block_summary, naive_summary),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / "budget_uncertainty.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  报告已写入 {out_json}")

    plot_forest(
        block_summary,
        point_estimate,
        IMAGES_DIR / "budget_ci.png",
        naive_summary=naive_summary,
    )

    _print_summary(report)
    return report


def _width_comparison(block: dict, naive: dict) -> dict:
    """逐渠道对比 block vs naive 的 CI 宽度，量化 block 多覆盖的不确定性。"""
    out = {"channels": {}}
    for ch, d in block["channels"].items():
        nd = naive["channels"].get(ch)
        if not nd:
            continue
        out["channels"][ch] = {
            "block_width": d["ci_width"],
            "naive_width": nd["ci_width"],
            "block_over_naive_ratio": (
                d["ci_width"] / nd["ci_width"] if nd["ci_width"] > 0 else None
            ),
        }
    # revenue lift 宽度对比
    bl, nl = block.get("revenue_lift_pct", {}), naive.get("revenue_lift_pct", {})
    if bl and nl:
        out["revenue_lift_pct"] = {
            "block_width": bl["ci_width"],
            "naive_width": nl["ci_width"],
            "block_over_naive_ratio": (
                bl["ci_width"] / nl["ci_width"] if nl["ci_width"] > 0 else None
            ),
        }
    return out


def _print_summary(report: dict) -> None:
    """控制台打印关键数字：哪些渠道 CI 窄/宽、block vs naive 对比。"""
    print("\n" + "=" * 70)
    print("关键结果")
    print("=" * 70)
    blk = report["block_bootstrap_ci"]["channels"]
    cmp = report["block_vs_naive_width"]["channels"]

    # 按 CI 宽度排序，最窄/最宽各列出来
    ordered = sorted(blk.items(), key=lambda kv: kv[1]["ci_width"])
    print("\n  各渠道最优 spend 的 95% CI（block bootstrap）：")
    for ch, d in ordered:
        ratio = cmp.get(ch, {}).get("block_over_naive_ratio")
        ratio_str = f"，block/naive={ratio:.2f}x" if ratio else ""
        print(
            f"    {ch:32s} 中位 ${d['median']:>10,.0f}  "
            f"CI [${d['ci_low']:>10,.0f}, ${d['ci_high']:>10,.0f}]"
            f"  宽 ${d['ci_width']:>10,.0f}{ratio_str}"
        )

    print(f"\n  最稳定（CI 最窄）：{ordered[0][0]}  宽 ${ordered[0][1]['ci_width']:,.0f}")
    print(f"  最不确定（CI 最宽）：{ordered[-1][0]}  宽 ${ordered[-1][1]['ci_width']:,.0f}")

    bl_lift = report["block_bootstrap_ci"].get("revenue_lift_pct", {})
    if bl_lift:
        print(
            f"\n  Revenue 提升点估计: {report['point_revenue']['improvement_pct']:.2f}%  "
            f"| block 95% CI: [{bl_lift['ci_low']:.2f}%, {bl_lift['ci_high']:.2f}%]"
        )

    # block vs naive 平均宽度比
    ratios = [v["block_over_naive_ratio"] for v in cmp.values() if v["block_over_naive_ratio"]]
    if ratios:
        print(
            f"\n  Block bootstrap CI 平均是 naive 的 "
            f"{np.mean(ratios):.2f}x 宽（保留自相关 → 更诚实的不确定性）"
        )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="预算优化不确定性：block bootstrap 置信区间")
    parser.add_argument(
        "-n",
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"bootstrap 次数（默认 {N_BOOTSTRAP}）",
    )
    parser.add_argument(
        "-b",
        "--block-size",
        type=int,
        default=BLOCK_SIZE_DAYS,
        help=f"block 大小（天，默认 {BLOCK_SIZE_DAYS}）",
    )
    args = parser.parse_args()
    run(n_bootstrap=args.n_bootstrap, block_size=args.block_size)


if __name__ == "__main__":
    main()
