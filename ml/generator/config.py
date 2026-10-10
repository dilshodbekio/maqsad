"""Generator configuration: static reference data and per-split parameters.

Everything that differs between the train and test portfolios lives in TRAIN / TEST
below, so the distribution shift is documented in one place.
"""
from __future__ import annotations

from dataclasses import dataclass

PURPOSES = ("tikuv_dastgohi", "chorva", "issiqxona", "parranda", "asalarichilik")
PURPOSE_WEIGHTS = (0.20, 0.28, 0.17, 0.20, 0.15)
# Median loan size in UZS (ASSUMPTION: illustrative, not taken from real bank data).
MEDIAN_AMOUNT = {
    "tikuv_dastgohi": 35_000_000,
    "chorva": 60_000_000,
    "issiqxona": 80_000_000,
    "parranda": 50_000_000,
    "asalarichilik": 30_000_000,
}

# (name, lat, lon) - approximate region centroids (ASSUMPTION: rounded, synthetic use only).
REGIONS = (
    ("Toshkent shahri", 41.31, 69.28),
    ("Toshkent viloyati", 41.20, 69.80),
    ("Samarqand", 39.65, 66.96),
    ("Buxoro", 39.77, 64.42),
    ("Farg'ona", 40.38, 71.78),
    ("Andijon", 40.78, 72.34),
    ("Namangan", 41.00, 71.67),
    ("Qashqadaryo", 38.86, 65.79),
    ("Surxondaryo", 37.94, 67.57),
    ("Xorazm", 41.55, 60.63),
    ("Navoiy", 40.10, 65.37),
    ("Jizzax", 40.12, 67.84),
    ("Sirdaryo", 40.49, 68.78),
    ("Qoraqalpog'iston", 43.80, 59.00),
)
REGION_WEIGHTS_TRAIN = (2, 4, 10, 6, 9, 8, 7, 9, 6, 5, 3, 4, 2, 5)
REGION_WEIGHTS_TEST = (3, 6, 8, 5, 7, 6, 9, 6, 9, 7, 4, 5, 3, 8)

# Illustrative OKED-style codes (ASSUMPTION: NOT checked against the official classifier).
MATCHING_OKED = {
    "tikuv_dastgohi": ("28.94", "46.64"),
    "chorva": ("01.41", "01.42", "46.23"),
    "issiqxona": ("01.13", "25.11", "46.73"),
    "parranda": ("01.47", "10.91", "46.21"),
    "asalarichilik": ("01.49", "46.22"),
}
GENERIC_OKED = ("46.90", "47.19", "47.89")
UNRELATED_OKED = ("56.10", "68.20", "62.01", "49.41", "96.09")

BRANCHES_PER_REGION = 2
EMPLOYEES_PER_BRANCH = 3
EMPLOYEE_ROLES = ("kredit_mutaxassisi", "monitoring_xodimi", "filial_boshligi")

T4_FRACTION_NAMES = ("T1", "T2", "T3", "T4")


@dataclass(frozen=True)
class Params:
    split: str
    seed: int
    snapshot_date: str
    id_offset: int
    n_loans: int = 5000
    # --- fraud mix ---
    fraud_rate: float = 0.11
    typology_mix: tuple = (0.30, 0.30, 0.20, 0.20)  # T1..T4
    p_stealth: float = 0.25                 # share of T1 / T2 that look "clean"
    # --- T1 ---
    t1_delay_mean_days: float = 0.7         # mean delay to first cash chunk (non-stealth)
    # --- T2 ---
    ring_extra_mean: float = 3.0            # ring size = 4 + Poisson(mean) (non-stealth)
    ring_employee_link: float = 0.40
    # --- T3 ---
    t3_max_day: float = 150.0               # resale happens U(20, t3_max_day) days after disbursement
    # --- honest ---
    honest_employee_link: float = 0.025
    honest_short_window_share: float = 0.18  # honest suppliers (>= 2 borrowers) active in a short window only
    honest_link_same_branch: float = 0.75   # linked employee works in a borrower's branch (rings: 0.80)
    honest_mismatch: float = 0.07           # share of honest loans with OKED mismatch
    seller_scale: float = 1.0               # scales honest livestock/poultry sellers (~5% of loans)
    cash_heavy_share: float = 0.05          # honest cash-heavy rural businesses
    # --- misc ---
    amount_scale: float = 1.0
    photo_reuse_fraud: float = 0.35
    photo_reuse_honest: float = 0.02
    photo_far_fraud: float = 0.25
    photo_far_honest: float = 0.04
    region_weights: tuple = REGION_WEIGHTS_TRAIN


TRAIN = Params(
    split="train", seed=42, snapshot_date="2026-03-31", id_offset=0,
)

# Test portfolio: different seed, different snapshot, disjoint ID space and slightly
# shifted parameters (harder: more stealth, slower cash-out, smaller rings, more
# honest-suspicious borrowers, bigger loans, different regional mix).
TEST = Params(
    split="test", seed=1337, snapshot_date="2026-07-31", id_offset=500_000,
    fraud_rate=0.13,
    p_stealth=0.35,
    t1_delay_mean_days=0.9,
    ring_extra_mean=2.5,
    ring_employee_link=0.33,
    t3_max_day=130.0,
    honest_mismatch=0.08,
    honest_link_same_branch=0.70,
    seller_scale=1.3,
    cash_heavy_share=0.065,
    amount_scale=1.10,
    region_weights=REGION_WEIGHTS_TEST,
)

SPLITS = {"train": TRAIN, "test": TEST}
