from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "baseline_v2_features.csv"
)

GT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "gt_boundaries.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "ml_boundary_dataset.csv"
)


# ============================================================
# Configuration
# ============================================================

LABEL_TOLERANCE_SECONDS = 5

RANDOM_SEED = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


# ============================================================
# Utility
# ============================================================

def fail(message):
    raise RuntimeError(f"\nML DATASET AUDIT FAILED:\n{message}\n")


# ============================================================
# Load feature table
# ============================================================

def load_features():
    if not FEATURE_FILE.exists():
        fail(
            f"Feature file does not exist:\n"
            f"{FEATURE_FILE}"
        )

    print("Loading V2 feature table...")
    df = pd.read_csv(FEATURE_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    return df


# ============================================================
# Audit basic structure
# ============================================================

def audit_structure(df):
    required_columns = {
        "session_id",
        "candidate_ts",
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(
            "Missing required columns:\n"
            + ", ".join(sorted(missing))
        )

    if df["session_id"].isna().any():
        fail("session_id contains missing values.")

    if df["candidate_ts"].isna().any():
        fail("candidate_ts contains missing values.")

    df["candidate_ts"] = pd.to_datetime(
        df["candidate_ts"],
        utc=True,
        errors="coerce",
    )

    if df["candidate_ts"].isna().any():
        fail(
            "Some candidate_ts values could not be "
            "parsed as UTC timestamps."
        )

    duplicate_count = df.duplicated(
        subset=["session_id", "candidate_ts"]
    ).sum()

    print(
        f"Duplicate candidate timestamps: "
        f"{duplicate_count:,}"
    )

    if duplicate_count:
        fail(
            f"Found {duplicate_count:,} duplicate "
            f"(session_id, candidate_ts) rows."
        )

    return df


# ============================================================
# Audit numeric features
# ============================================================

def audit_numeric_features(df):
    excluded = {
        "session_id",
        "candidate_ts",
    }

    feature_columns = [
        c for c in df.columns
        if c not in excluded
    ]

    non_numeric = []

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            non_numeric.append(column)

    if non_numeric:
        print()
        print("Non-numeric columns:")
        for column in non_numeric:
            print(f"  - {column}")

        fail(
            "All ML features must currently be numeric. "
            "Unexpected non-numeric columns were found."
        )

    numeric_df = df[feature_columns]

    nan_counts = numeric_df.isna().sum()

    nan_features = nan_counts[
        nan_counts > 0
    ]

    print(
        f"Rows containing NaN values: "
        f"{int(numeric_df.isna().any(axis=1).sum()):,}"
    )

    if len(nan_features):
        print()
        print("NaN counts by feature:")
        print(nan_features.to_string())

        fail(
            "NaN values exist in ML features. "
            "We should fix them rather than silently filling them."
        )

    inf_mask = np.isinf(
        numeric_df.to_numpy()
    )

    inf_count = int(inf_mask.sum())

    print(
        f"Infinite feature values: "
        f"{inf_count:,}"
    )

    if inf_count:
        fail(
            "Infinite values exist in ML features."
        )

    # Constant columns are not necessarily fatal,
    # but they are not useful to the model.
    constant_columns = [
        column
        for column in feature_columns
        if numeric_df[column].nunique() <= 1
    ]

    print(
        f"Constant feature columns: "
        f"{len(constant_columns):,}"
    )

    if constant_columns:
        print()
        for column in constant_columns:
            print(f"  - {column}")

    return df, feature_columns


# ============================================================
# Check for GT leakage
# ============================================================

def audit_leakage(df):
    forbidden_exact = {
        "process_code",
        "process_name",
        "process_variant",
        "current_process",
        "case_id",
        "gt_ts",
        "boundary_ts",
        "boundary_event",
        "boundary_from",
        "boundary_to",
        "label",
    }

    suspicious = []

    for column in df.columns:
        normalized = column.lower()

        if column in forbidden_exact:
            suspicious.append(column)
            continue

        # Don't reject legitimate words such as
        # "before_event_count". Only catch obvious GT fields.
        suspicious_tokens = [
            "ground_truth",
            "gt_boundary",
            "process_code",
            "process_name",
            "current_process",
            "case_id",
        ]

        if any(
            token in normalized
            for token in suspicious_tokens
        ):
            suspicious.append(column)

    print()
    print(
        "Potential GT leakage columns: "
        f"{len(suspicious):,}"
    )

    if suspicious:
        print()
        for column in suspicious:
            print(f"  - {column}")

        fail(
            "Potential ground-truth leakage detected."
        )


# ============================================================
# Load GT
# ============================================================

def load_gt():
    if not GT_FILE.exists():
        fail(
            f"Ground-truth boundary file does not exist:\n"
            f"{GT_FILE}"
        )

    print()
    print("Loading ground-truth boundaries...")

    gt = pd.read_csv(GT_FILE)

    required = {
        "session_id",
        "ts",
    }

    missing = required - set(gt.columns)

    if missing:
        fail(
            "GT file is missing:\n"
            + ", ".join(sorted(missing))
        )

    gt["ts"] = pd.to_datetime(
        gt["ts"],
        utc=True,
        errors="coerce",
    )

    if gt["ts"].isna().any():
        fail(
            "Some ground-truth timestamps "
            "could not be parsed."
        )

    raw_count = len(gt)

    # Same normalization used by our current evaluation.
    gt = (
        gt[
            ["session_id", "ts"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Raw GT boundary events: "
        f"{raw_count:,}"
    )

    print(
        f"Unique GT timestamps used for evaluation: "
        f"{len(gt):,}"
    )

    return gt


# ============================================================
# Label candidates
# ============================================================

def create_labels(features, gt):
    """
    Label a candidate as positive when it falls within
    LABEL_TOLERANCE_SECONDS of a GT boundary.

    This uses only Dataset A ground truth to construct
    training labels. GT identifiers themselves are never
    included as model features.
    """

    print()
    print(
        "Creating boundary/non-boundary labels..."
    )

    gt_by_session = {}

    for session_id, group in gt.groupby(
        "session_id"
    ):
        gt_by_session[session_id] = (
            group["ts"]
            .sort_values()
            .tolist()
        )

    labels = []

    tolerance = pd.Timedelta(
        seconds=LABEL_TOLERANCE_SECONDS
    )

    for _, row in features.iterrows():

        session_id = row["session_id"]
        candidate_ts = row["candidate_ts"]

        session_gt = gt_by_session.get(
            session_id,
            [],
        )

        if not session_gt:
            labels.append(0)
            continue

        is_boundary = any(
            abs(candidate_ts - gt_ts)
            <= tolerance
            for gt_ts in session_gt
        )

        labels.append(
            1 if is_boundary else 0
        )

    features = features.copy()

    features["label"] = labels

    return features


# ============================================================
# Session-level split
# ============================================================

def assign_session_splits(df):
    """
    Split whole sessions rather than individual rows.
    """

    print()
    print(
        "Creating session-level train/validation/test split..."
    )

    sessions = sorted(
        df["session_id"].unique()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rng.shuffle(sessions)

    n_sessions = len(sessions)

    n_train = int(
        n_sessions * TRAIN_FRACTION
    )

    n_validation = int(
        n_sessions * VALIDATION_FRACTION
    )

    train_sessions = sessions[
        :n_train
    ]

    validation_sessions = sessions[
        n_train:
        n_train + n_validation
    ]

    test_sessions = sessions[
        n_train + n_validation:
    ]

    split_map = {}

    for session in train_sessions:
        split_map[session] = "train"

    for session in validation_sessions:
        split_map[session] = "validation"

    for session in test_sessions:
        split_map[session] = "test"

    df = df.copy()

    df["split"] = (
        df["session_id"]
        .map(split_map)
    )

    if df["split"].isna().any():
        fail(
            "Some sessions were not assigned "
            "to a split."
        )

    # Verify there is no session leakage.
    train_set = set(train_sessions)
    validation_set = set(validation_sessions)
    test_set = set(test_sessions)

    if train_set & validation_set:
        fail(
            "Session leakage between train and validation."
        )

    if train_set & test_set:
        fail(
            "Session leakage between train and test."
        )

    if validation_set & test_set:
        fail(
            "Session leakage between validation and test."
        )

    print(
        f"Train sessions:      {len(train_sessions):,}"
    )
    print(
        f"Validation sessions: {len(validation_sessions):,}"
    )
    print(
        f"Test sessions:       {len(test_sessions):,}"
    )

    return df


# ============================================================
# Dataset statistics
# ============================================================

def print_statistics(df):
    print()
    print("=" * 60)
    print("ML DATASET SUMMARY")
    print("=" * 60)

    print()
    print("Total rows:")
    print(f"  {len(df):,}")

    print()
    print("Total sessions:")
    print(
        f"  {df['session_id'].nunique():,}"
    )

    print()
    print("Labels:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "non-boundary",
                1: "boundary",
            }
        )
        .to_string()
    )

    print()
    print("Overall boundary rate:")
    print(
        f"  {df['label'].mean() * 100:.2f}%"
    )

    print()
    print("Rows by split:")

    print(
        df["split"]
        .value_counts()
        .to_string()
    )

    print()
    print("Labels by split:")

    split_label_counts = (
        df.groupby(
            ["split", "label"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(
        split_label_counts.to_string()
    )

    print()
    print("Sessions by split:")

    session_split_counts = (
        df.groupby("split")[
            "session_id"
        ]
        .nunique()
    )

    print(
        session_split_counts.to_string()
    )


# ============================================================
# Save
# ============================================================

def save_dataset(df, feature_columns):
    """
    Save the ML dataset.

    Keep session_id/candidate_ts/split for evaluation and
    reproducibility. The model-training code can select only
    feature_columns.
    """

    output_columns = (
        [
            "session_id",
            "candidate_ts",
            "label",
            "split",
        ]
        + feature_columns
    )

    # Remove duplicate columns while preserving order.
    output_columns = list(
        dict.fromkeys(output_columns)
    )

    df[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Saved ML dataset:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Rows written: {len(df):,}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("BUILD ML DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load + audit features
    # --------------------------------------------------------

    features = load_features()

    features = audit_structure(
        features
    )

    features, feature_columns = (
        audit_numeric_features(
            features
        )
    )

    audit_leakage(
        features
    )

    # --------------------------------------------------------
    # 2. Load GT
    # --------------------------------------------------------

    gt = load_gt()

    # --------------------------------------------------------
    # 3. Check session coverage
    # --------------------------------------------------------

    feature_sessions = set(
        features["session_id"]
    )

    gt_sessions = set(
        gt["session_id"]
    )

    missing_gt_sessions = (
        feature_sessions
        - gt_sessions
    )

    if missing_gt_sessions:
        print()
        print(
            "Sessions with features but no GT:"
        )

        for session in sorted(
            missing_gt_sessions
        ):
            print(
                f"  - {session}"
            )

        print(
            "\nThis is not automatically fatal, "
            "but for Dataset A we expect GT coverage."
        )

    # --------------------------------------------------------
    # 4. Create labels
    # --------------------------------------------------------

    dataset = create_labels(
        features,
        gt,
    )

    # --------------------------------------------------------
    # 5. Split by session
    # --------------------------------------------------------

    dataset = assign_session_splits(
        dataset
    )

    # --------------------------------------------------------
    # 6. Statistics
    # --------------------------------------------------------

    print_statistics(
        dataset
    )

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    save_dataset(
        dataset,
        feature_columns,
    )

    print()
    print("=" * 60)
    print("AUDIT PASSED")
    print("ML dataset is ready for model comparison.")
    print("=" * 60)


if __name__ == "__main__":
    main()