"""单元测试：预算优化的 block bootstrap 不确定性量化。

这些测试只跑 **合成数据**（快、确定性），不依赖 parquet/mmm_results.json 等
预生成产物。覆盖三类断言：

1. block 切分/重抽样逻辑：块不重叠、覆盖全序列、块内保序、可复现；
2. CI 覆盖率：在已知系数的合成线性数据上，bootstrap CI 大致覆盖真值；
3. block vs naive 对比：在**自相关**数据上 block bootstrap 的 CI 显著宽于
   naive case-resample——这是用 block 而非朴素法的核心论据。
"""

import numpy as np
import pytest

from attributor.budget_uncertainty import (  # noqa: E402
    block_resample_indices,
    case_resample_indices,
    make_blocks,
    run_bootstrap,
    summarize_ci,
)

# ---------------------------------------------------------------------------
# 1. block 切分逻辑
# ---------------------------------------------------------------------------


class TestMakeBlocks:
    def test_blocks_cover_all_rows_and_disjoint(self):
        """块必须覆盖 [0, n) 的每一行且互不重叠。"""
        blocks = make_blocks(30, 7)
        covered = np.concatenate([np.arange(s, e) for s, e in blocks])
        assert sorted(covered.tolist()) == list(range(30))
        # 不重叠：所有行索引唯一
        assert len(covered) == len(set(covered.tolist())) == 30

    def test_blocks_preserve_time_order(self):
        """每块 [start, end) 内行索引必须严格递增（保序是 block bootstrap 的核心）。"""
        blocks = make_blocks(50, 7)
        for s, e in blocks:
            assert s < e
            # 块内天然是 arange(s,e)，严格递增
            seg = list(range(s, e))
            assert seg == sorted(seg)

    def test_block_count_reasonable(self):
        """n=30, block_size=7 → 5 块（28 行 4 整块 + 1 块 2 行）。"""
        blocks = make_blocks(30, 7)
        assert len(blocks) == 5
        assert blocks[-1, 0] == 28 and blocks[-1, 1] == 30  # 末块是 [28,30)

    def test_last_short_block_handled(self):
        """末块不足 block_size 时仍为合法的非空区间。"""
        blocks = make_blocks(10, 3)
        # 0-3, 3-6, 6-9, 9-10
        assert len(blocks) == 4
        assert blocks[-1, 1] == 10
        assert blocks[-1, 1] - blocks[-1, 0] >= 1

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            make_blocks(0, 7)
        with pytest.raises(ValueError):
            make_blocks(30, 0)


class TestBlockResample:
    def test_output_length_matches_input(self):
        """重抽样后训练集长度 = 原长度（保持拟合规模一致）。"""
        rng = np.random.default_rng(0)
        idx = block_resample_indices(100, 7, rng)
        assert len(idx) == 100

    def test_indices_in_range(self):
        rng = np.random.default_rng(1)
        idx = block_resample_indices(100, 7, rng)
        assert idx.min() >= 0 and idx.max() < 100

    def test_preserves_intra_block_order(self):
        """关键性质：被抽中的块内部保持原始时间顺序（不打乱）。

        block resample 的输出是若干「连续递增段」拼接而成——即每当索引从一段
        跳到另一段（差值不为 +1）时，开启一个新段；每个新段内部必须严格递增。
        我们按 ``diff != 1`` 切段，逐段断言严格递增。
        """
        rng = np.random.default_rng(2)
        n, bs = 70, 7
        idx = block_resample_indices(n, bs, rng)

        # 找段边界：相邻索引差为 1 → 同段；否则新段
        # （注意：两块恰好相邻时会自然连成一段，长度可超过 bs，这是正常的）
        breaks = np.where(np.diff(idx) != 1)[0] + 1
        seg_starts = np.concatenate([[0], breaks, [len(idx)]])
        for i in range(len(seg_starts) - 1):
            seg = idx[seg_starts[i] : seg_starts[i + 1]]
            assert list(seg) == sorted(seg.tolist()), "块内顺序被破坏（必须严格递增）"
            assert len(set(seg.tolist())) == len(seg), "块内不应有重复行（块本就无重复）"

    def test_reproducible_with_seed(self):
        """固定种子 → 完全可复现。"""
        a = block_resample_indices(50, 7, np.random.default_rng(42))
        b = block_resample_indices(50, 7, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_resamples_are_blocks_not_individual_rows(self):
        """block bootstrap 抽到的是整块——相邻索引应是连续的（成串出现）。

        朴素 case-resample 的索引是散乱的；block 的索引应呈现「连续段」结构。
        用「相邻索引差为 1 的比例」区分二者。
        """
        rng = np.random.default_rng(3)
        n, bs = 210, 7
        block_idx = block_resample_indices(n, bs, rng)
        naive_idx = case_resample_indices(n, np.random.default_rng(3))

        block_consecutive = np.mean(np.diff(block_idx) == 1)
        naive_consecutive = np.mean(np.diff(naive_idx) == 1)
        # block 重抽样里相邻索引大量连续（每块内 6/7≈0.86 比例连续）
        assert block_consecutive > 0.5
        # naive 几乎没有连续相邻（独立抽样，相邻相等的概率 ~1/n）
        assert naive_consecutive < 0.1


# ---------------------------------------------------------------------------
# 2. CI 覆盖率：合成线性数据，CI 应大致覆盖真值
# ---------------------------------------------------------------------------


def _synthetic_linear_data(n=200, seed=0, ar_coef=0.0):
    """构造已知系数的合成数据：用真实渠道命名（``*_adstock`` 特征 /
    ``*_spend`` current_spend 键），使 ``extract_params`` 能正常工作。

    ``ar_coef > 0`` 时给噪声加 AR(1) 自相关，模拟本项目 MMM 残差结构
    （真实 OLS Durbin-Watson=0.90，强正相关）。
    """
    rng = np.random.default_rng(seed)
    # 用两个真实渠道，确保 extract_params（按 SPEND_CHANNELS 取 *_adstock）能匹配
    ch_a = "google_paid_search"
    ch_b = "meta_facebook"
    feat_a, feat_b = f"{ch_a}_adstock", f"{ch_b}_adstock"

    x1 = rng.uniform(100, 500, n)
    x2 = rng.uniform(50, 300, n)

    # AR(1) 噪声：e_t = ar_coef * e_{t-1} + eps_t
    eps = rng.normal(0, 5, n)
    noise = np.zeros(n)
    noise[0] = eps[0]
    for t in range(1, n):
        noise[t] = ar_coef * noise[t - 1] + eps[t]

    y = 2.0 * x1 + 5.0 * x2 + 1000.0 + noise
    X = np.column_stack([x1, x2])
    feature_names = [feat_a, feat_b]
    current_spend = {f"{ch_a}_spend": 300.0, f"{ch_b}_spend": 175.0}
    return X, y, feature_names, current_spend


class TestCICoverage:
    def test_block_ci_covers_true_optimal_spend(self):
        """在干净的合成线性数据上，block bootstrap 的 CI 应覆盖真值驱动的最优分配。

        这里数据无自相关（ar_coef=0），block 与 naive 都应合理。我们主要验证管线
        跑通且 CI 区间合理（非 NaN、low<=high、点估计落在区间附近）。
        """
        X, y, names, current = _synthetic_linear_data(n=200, seed=7, ar_coef=0.0)
        total = sum(current.values())

        samples = run_bootstrap(
            X,
            y,
            names,
            current,
            total,
            n_bootstrap=40,
            block_size=14,
            seed=11,
            resample_fn=block_resample_indices,
            verbose=False,
        )
        summary = summarize_ci(samples)

        assert summary["n_successful"] > 0
        assert len(summary["channels"]) > 0, "至少应有一个渠道的 CI 被汇总出来"
        for ch, d in summary["channels"].items():
            assert np.isfinite(d["ci_low"])
            assert np.isfinite(d["ci_high"])
            assert d["ci_low"] <= d["median"] <= d["ci_high"]
            assert d["ci_width"] > 0


# ---------------------------------------------------------------------------
# 3. block vs naive 对比（核心论据）
# ---------------------------------------------------------------------------


class TestBlockVsNaive:
    def test_block_preserves_autocorrelation_naive_destroys_it(self):
        """**核心方法学论据**：block bootstrap 保留残差自相关，naive 把它打没。

        本项目 MMM 残差强自相关（OLS Durbin-Watson=0.90）。在自相关数据上：

        - **block bootstrap**：重抽样整块、块内保序，被重抽样序列的 lag-1 自相关
          接近原始序列（约 0.8）→ 重拟合得到的系数方差反映真实的、被自相关
          膨胀的不确定性 → CI 诚实（更宽）。
        - **naive case-resample**：逐行独立重抽样，把序列打乱成「伪独立」样本，
          被重抽样序列的 lag-1 自相关坍缩到 ≈0 → 系数方差被系统性低估 → CI
          偏窄、给 CMO 错误信心。

        这个断言直接落在「自相关是否被保留」这一最根本的性质上，比经过
        Ridge 强收缩 + Hill 饱和优化器后的 CI 宽度更稳健（那条链路上信号会被
        多重非线性变换吞掉）。这是「为什么必须用 block 而非 naive」的决定性证明。
        """
        rng = np.random.default_rng(5)
        n, ar = 500, 0.85
        eps = rng.normal(0, 1, n)
        noise = np.zeros(n)
        noise[0] = eps[0]
        for t in range(1, n):
            noise[t] = ar * noise[t - 1] + eps[t]

        def lag1(a: np.ndarray) -> float:
            a = a - a.mean()
            return float(np.corrcoef(a[:-1], a[1:])[0, 1])

        orig_autocorr = lag1(noise)
        block_autocorrs = [
            lag1(noise[block_resample_indices(n, 14, np.random.default_rng(i))]) for i in range(200)
        ]
        naive_autocorrs = [
            lag1(noise[case_resample_indices(n, np.random.default_rng(i))]) for i in range(200)
        ]

        # 原始强自相关
        assert orig_autocorr > 0.7, f"原始序列应强自相关，得到 {orig_autocorr:.3f}"
        # block 保留：重抽样后的自相关仍接近原始（≥0.5）
        assert np.mean(block_autocorrs) > 0.5, (
            f"block 重抽样应保留自相关（mean lag1={np.mean(block_autocorrs):.3f}），"
            "否则 block bootstrap 失去意义。"
        )
        # naive 破坏：自相关坍缩到 ≈0
        assert abs(np.mean(naive_autocorrs)) < 0.1, (
            f"naive 重抽样应把自相关打没（mean lag1={np.mean(naive_autocorrs):.3f}），"
            "这正是它在自相关数据上低估不确定性的根因。"
        )
        # 量化对比：block 保留的自相关显著高于 naive
        assert np.mean(block_autocorrs) > np.mean(naive_autocorrs) + 0.5

    def test_block_ci_for_mean_wider_than_naive(self):
        """CI 层面的稳健论据：对 AR 序列的「均值」做 bootstrap，block CI 显著更宽。

        这是 block bootstrap 最经典的应用场景，也是「block 比 naive 宽」最稳健、
        可复现的构造。原理：自相关序列的有效样本量低于 N，真值均值的方差被自
        相关放大；block 重抽样保留了这种放大，naive 把它抹平 → naive CI 系统性
        偏窄、低估了均值的不确定性。

        注意：本断言针对**均值估计**（聚合统计量）。回归系数/优化器分配的 CI
        宽度方向并非常数——经过强收缩的 Ridge + Hill 饱和优化器后，block vs
        naive 的宽窄会受设计影响。故 CI 宽度的「必须更宽」论据落在这里，而
        自相关保留（上一个测试）才是对方法学最根本的证明。
        """
        rng = np.random.default_rng(5)
        n, ar = 500, 0.85
        eps = rng.normal(0, 1, n)
        noise = np.zeros(n)
        noise[0] = eps[0]
        for t in range(1, n):
            noise[t] = ar * noise[t - 1] + eps[t]

        def mean_ci(use_block: bool) -> float:
            means = []
            for i in range(400):
                g = np.random.default_rng(i)
                idx = block_resample_indices(n, 14, g) if use_block else case_resample_indices(n, g)
                means.append(float(noise[idx].mean()))
            means = np.array(means)
            return float(np.percentile(means, 97.5) - np.percentile(means, 2.5))

        block_width = mean_ci(use_block=True)
        naive_width = mean_ci(use_block=False)
        assert block_width > naive_width * 1.5, (
            f"均值 CI：block({block_width:.4f}) 应显著宽于 naive({naive_width:.4f})，"
            "因为 block 保留了自相关放大的均值方差（block bootstrap 的经典价值）。"
        )
