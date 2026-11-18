"""Column roles. Every model input must come from CAT_INPUTS or NUM_INPUTS.

Labels other than the target are kept only as `label_*` columns for later auxiliary
experiments and are never model inputs (checked by `assert_no_label_inputs`).
"""

PAD = "dbee74ee5fbf"  # [추정] missing/padding hash token (docs/DESIGN.md §10.1 E5)

TARGET = "last"  # y = (last > 0): last-click attributed conversion
AUX_LABELS = ["first", "linear", "mta"]
RAW_LABELS = [TARGET] + AUX_LABELS

USER_COLS = [f"user_feat{i}" for i in range(6)]
CTX_COLS = ["context_feat1", "context_feat2", "context_feat3"]  # context_feat0: ~2.2M values/day, unused
AD_COLS = ["ad_feat1", "ad_feat2", "ad_feat3", "ad_feat4", "ad_feat5"]  # ad_feat0 (item) used via history only
VOCAB_COLS = USER_COLS + CTX_COLS + AD_COLS

SEQ_PAIRS = [("ad_feat0", "ad_feat0_seq"), ("ad_feat1", "ad_feat1_seq"), ("ad_feat2", "ad_feat2_seq")]
MM_COLS = ["mm_feat0_seq", "mm_feat1_seq"]

GROUP_KEY = "ad_feat1"  # [추정] category-like anonymous item feature
SHOP_KEY = "ad_feat2"  # [추정] shop-like anonymous item feature
DOMAIN_KEY = "context_feat1"  # study C domain: anonymous ad-context groups (docs/DESIGN.md §16), not asserted to be scenarios

# integer codes reserved by the vocabulary
UNKNOWN = 0  # value not seen in the training files
MISSING = 1  # the PAD token

# M1 and M2 use the SAME inputs: CAT_INPUTS + NUM_INPUTS. The *_nohist variants drop NUM_INPUTS.
CAT_INPUTS = USER_COLS + CTX_COLS + ["ad_feat1", SHOP_KEY, "ad_feat3", "ad_feat4", "ad_feat5"]
NUM_INPUTS = [
    "match_ad_feat0", "match_ad_feat1", "match_ad_feat2", "hist_empty",
    "mm_feat0_max", "mm_feat0_mean", "mm_feat1_max", "mm_feat1_mean",
]
HISTORY_INPUTS = NUM_INPUTS  # all numeric inputs come from the purchase history
LR_ONEHOT = CAT_INPUTS


def assert_no_label_inputs(inputs: list[str]) -> None:
    bad = [c for c in inputs if c.startswith("label_") or c in RAW_LABELS or c == "y"]
    if bad:
        raise ValueError(f"label columns used as inputs: {bad}")
