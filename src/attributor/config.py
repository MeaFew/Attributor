"""Marketing Attribution & Budget Optimization - Centralized Configuration."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (resolved from project root, two parents up from src/attributor/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
RAW_CSV_PATH = RAW_DATA_DIR / "conjura_mmm_data.csv"
DATA_DICT_PATH = RAW_DATA_DIR / "conjura_mmm_data_dictionary.tsv"
CLEANED_PARQUET_PATH = PROCESSED_DATA_DIR / "mmm_cleaned.parquet"

# Simulated touchpoint data
SIMULATED_TOUCHPOINTS_PATH = PROCESSED_DATA_DIR / "simulated_touchpoints.parquet"
SIMULATED_JOURNEYS_PATH = PROCESSED_DATA_DIR / "simulated_journeys.parquet"

# Real Criteo attribution data
CRITEO_RAW_PATH = RAW_DATA_DIR / "criteo_attribution_dataset.tsv.gz"
CRITEO_TOUCHPOINTS_PATH = PROCESSED_DATA_DIR / "criteo_touchpoints.parquet"
CRITEO_JOURNEYS_PATH = PROCESSED_DATA_DIR / "criteo_journeys.parquet"

# Output directories
MODEL_OUTPUT_DIR = PROCESSED_DATA_DIR / "models"


def ensure_dirs() -> None:
    """Create the project's output directories. Call once at startup.

    Side-effect-free import: merely importing ``config`` no longer creates
    directories on disk. Every writing script already mkdir's its own output
    parent, and pipeline entrypoints call this explicitly, so removing the
    import-time side effect makes ``config`` safe to import read-only (e.g. in
    tests / CI / the dashboard) without mutating the filesystem.
    """
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, IMAGES_DIR, MODEL_OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
# Spend channels (from figshare data)
SPEND_CHANNELS = [
    "google_paid_search_spend",
    "google_shopping_spend",
    "google_pmax_spend",
    "google_display_spend",
    "google_video_spend",
    "meta_facebook_spend",
    "meta_instagram_spend",
    "meta_other_spend",
    "tiktok_spend",
]

CLICK_CHANNELS = [
    "google_paid_search_clicks",
    "google_shopping_clicks",
    "google_pmax_clicks",
    "google_display_clicks",
    "google_video_clicks",
    "meta_facebook_clicks",
    "meta_instagram_clicks",
    "meta_other_clicks",
    "tiktok_clicks",
]

IMPRESSION_CHANNELS = [
    "google_paid_search_impressions",
    "google_shopping_impressions",
    "google_pmax_impressions",
    "google_display_impressions",
    "google_video_impressions",
    "meta_facebook_impressions",
    "meta_instagram_impressions",
    "meta_other_impressions",
    "tiktok_impressions",
]

# Organic / non-paid channels (clicks only)
ORGANIC_CHANNELS = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
    "all_other_clicks",
]

# Target variables
TARGET_NEW_CUSTOMERS = "first_purchases"
TARGET_NEW_REVENUE = "first_purchases_original_price"
TARGET_ALL_CUSTOMERS = "all_purchases"
TARGET_ALL_REVENUE = "all_purchases_original_price"

# ---------------------------------------------------------------------------
# Modeling hyperparameters
# ---------------------------------------------------------------------------
# Centralized so README/docs and the code cannot drift apart. Each is documented
# inline where it is used.
HOLDOUT_FRACTION = 0.2  # time-based train/test split for MMM evaluation

# Adstock decay rate (preprocess.py): spend is decayed across lagged days with
# weights decay^k. 0.5 → moderate carryover.
ADSTOCK_DECAY = 0.5

# Time-decay attribution half-life (multi_touch_attribution.py), in days. A
# touchpoint half_life_days before conversion gets half the weight of the
# converting touchpoint.
ATTRIBUTION_HALF_LIFE_DAYS = 7.0

# Hill saturation slope (budget_optimizer.py): gamma > 1 gives an S-curve with
# mild saturation near the half-saturation point (tau = current spend).
HILL_GAMMA = 1.5

# ---------------------------------------------------------------------------
# 预算优化不确定性（block bootstrap）—— scripts/budget_uncertainty.py
# ---------------------------------------------------------------------------
# 单点估计对 CMO 砍预算很危险：没有不确定性量化。这里用 block bootstrap
# 给每个渠道的「最优 spend」与「revenue lift」配 95% 置信区间。
#
# 为什么是 *block* bootstrap 而不是朴素 case-resample：MMM 残差强自相关
# （本项目 OLS 的 Durbin-Watson=0.90，远小于 2.0）。朴素重抽样会打乱时序，
# 等于假设样本独立同分布，系统性低估不确定性、CI 偏窄。block bootstrap 按
# 时间分块、块内保持原始顺序地重抽样，保留块内自相关结构，得到诚实的不
# 确定性区间。详见 scripts/budget_uncertainty.py。
BLOCK_SIZE_DAYS = 7  # 连续时间块长度（天）；块内保留自相关
N_BOOTSTRAP = 200  # bootstrap 重抽样次数（每次重拟合 Ridge + 重跑优化）
BOOTSTRAP_CI_LEVEL = 0.95  # 置信区间水平（百分位法 [2.5%, 97.5%]）
BOOTSTRAP_RANDOM_SEED = 42  # 固定种子保证结果可复现

# Simulation parameters for touchpoint data
SIMULATION_PARAMS = {
    "n_users": 50_000,
    "max_touchpoints_per_user": 8,
    "conversion_rate": 0.035,
    "channels": [
        "google_paid_search",
        "google_shopping",
        "google_pmax",
        "google_display",
        "google_video",
        "meta_facebook",
        "meta_instagram",
        "meta_other",
        "tiktok",
        "direct",
        "branded_search",
        "organic_search",
        "email",
        "referral",
    ],
    "channel_weights": {
        "google_paid_search": 0.18,
        "google_shopping": 0.12,
        "google_pmax": 0.10,
        "google_display": 0.08,
        "google_video": 0.06,
        "meta_facebook": 0.15,
        "meta_instagram": 0.10,
        "meta_other": 0.03,
        "tiktok": 0.08,
        "direct": 0.03,
        "branded_search": 0.02,
        "organic_search": 0.02,
        "email": 0.02,
        "referral": 0.01,
    },
    "date_range_days": 365,
}
