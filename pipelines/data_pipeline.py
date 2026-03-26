"""
Shared Data Pipeline
====================
Common data loading, cleaning, feature engineering, and splitting logic
used across all models.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ── Discharge dispositions where readmission is impossible ──
EXCLUDE_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

# ── Age bracket → midpoint mapping ──
AGE_MAP = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

# ── Medication columns ──
MED_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

MED_ORDINAL_MAP = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}

# ── NLP effectiveness mapping (5 → 3 classes) ──
EFFECTIVENESS_MAP = {
    "Highly Effective": "Highly Effective",
    "Considerably Effective": "Somewhat Effective",
    "Moderately Effective": "Somewhat Effective",
    "Marginally Effective": "Ineffective",
    "Ineffective": "Ineffective",
}

# ── Numeric & categorical feature definitions ──
NUMERIC_FEATURES = [
    "age_numeric", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
    "num_med_changes", "num_active_meds",
    "total_prior_visits", "has_prior_inpatient", "has_prior_emergency",
    "inpatient_squared", "inpatient_log", "high_utilizer",
    "emergency_inpatient_interaction", "outpatient_inpatient_ratio",
    "lab_procedure_ratio", "total_procedures",
    "high_med_burden", "med_count_log", "high_diagnosis_count",
    "A1C_not_tested", "A1C_abnormal", "glu_not_tested", "glu_abnormal",
    "high_risk_discharge",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "inpatient_tier",
]


# ═══════════════════════════════════════════════════════════
# ICD-9 Diagnosis Code Grouping
# ═══════════════════════════════════════════════════════════

def group_diag_code(code):
    """Group an ICD-9 code into a clinical category."""
    if pd.isna(code) or str(code).strip() in ("?", ""):
        return "Unknown"
    code = str(code).strip()
    if code.startswith("V"):
        return "Supplemental"
    if code.startswith("E"):
        return "External"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 250 <= num < 251:
        return "Diabetes"
    if 390 <= num < 460:
        return "Circulatory"
    if 460 <= num < 520:
        return "Respiratory"
    if 520 <= num < 580:
        return "Digestive"
    if 580 <= num < 630:
        return "Genitourinary"
    if 710 <= num < 740:
        return "Musculoskeletal"
    if 800 <= num < 1000:
        return "Injury"
    if 140 <= num < 240:
        return "Neoplasms"
    if 240 <= num < 280:
        return "Endocrine"
    if 280 <= num < 290:
        return "Blood"
    if 290 <= num < 320:
        return "Mental"
    if 320 <= num < 390:
        return "Nervous"
    if 680 <= num < 710:
        return "Skin"
    return "Other"


# ═══════════════════════════════════════════════════════════
# Encounter Data Preprocessing
# ═══════════════════════════════════════════════════════════

def load_raw_encounters(filepath=None):
    """Load raw patient encounters CSV."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "healthcare_csvs" / "patient_encounters_2023.csv"
    return pd.read_csv(filepath)


def clean_encounters(df, filter_deceased=True, create_target=True):
    """Clean raw encounter data.

    Args:
        df: Raw encounter DataFrame
        filter_deceased: If True, remove patients who expired/hospice
        create_target: If True, create binary readmitted target

    Returns:
        Cleaned DataFrame with engineered features
    """
    df = df.copy()

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Drop columns that are mostly missing or not useful
    drop_cols = ["weight", "payer_code", "examide", "citoglipton"]
    df.drop(columns=[c for c in drop_cols if c in df.columns],
            inplace=True, errors="ignore")

    # Keep medical_specialty - fill missing with 'Unknown'
    if "medical_specialty" in df.columns:
        df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
        # Group rare specialties
        spec_counts = df["medical_specialty"].value_counts()
        rare_specs = spec_counts[spec_counts < 100].index
        df["medical_specialty"] = df["medical_specialty"].replace(
            {s: "Other" for s in rare_specs}
        )

    # Filter deceased/hospice patients (can't be readmitted)
    if filter_deceased and "discharge_disposition_id" in df.columns:
        df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)]

    # Convert age brackets to numeric
    if "age" in df.columns:
        df["age_numeric"] = df["age"].map(AGE_MAP).fillna(55)
        df.drop(columns=["age"], inplace=True)

    # Group diagnosis codes
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            df[f"{diag_col}_group"] = df[diag_col].apply(group_diag_code)
            df.drop(columns=[diag_col], inplace=True)

    # Encode medication columns ordinally
    active_med_cols = [c for c in MED_COLUMNS if c in df.columns]
    for col in active_med_cols:
        df[col] = df[col].map(MED_ORDINAL_MAP).fillna(0).astype(int)

    # Engineered features
    if active_med_cols:
        df["num_med_changes"] = (df[active_med_cols] >= 2).sum(axis=1)
        df["num_active_meds"] = (df[active_med_cols] > 0).sum(axis=1)

    # Prior utilization features
    for col in ["number_outpatient", "number_emergency", "number_inpatient"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if all(c in df.columns for c in ["number_outpatient", "number_emergency", "number_inpatient"]):
        df["total_prior_visits"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )
        df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
        df["has_prior_emergency"] = (df["number_emergency"] > 0).astype(int)
        # Non-linear prior inpatient (strongest single predictor)
        df["inpatient_squared"] = df["number_inpatient"] ** 2
        df["inpatient_log"] = np.log1p(df["number_inpatient"])
        df["high_utilizer"] = (df["number_inpatient"] >= 3).astype(int)
        df["emergency_inpatient_interaction"] = (
            df["number_emergency"] * df["number_inpatient"]
        )
        # Inpatient risk tiers (captures non-linear jump in readmission rates)
        df["inpatient_tier"] = pd.cut(
            df["number_inpatient"],
            bins=[-1, 0, 1, 2, 100],
            labels=["none", "low", "medium", "high"],
        ).astype(str)
        # Total prior utilization interaction
        df["outpatient_inpatient_ratio"] = (
            df["number_outpatient"] / (df["number_inpatient"] + 1)
        )

    # Lab/procedure intensity
    if all(c in df.columns for c in ["num_lab_procedures", "num_procedures"]):
        df["lab_procedure_ratio"] = df["num_lab_procedures"] / (df["num_procedures"] + 1)
        df["total_procedures"] = df["num_lab_procedures"] + df["num_procedures"]

    # Medication burden
    if "num_medications" in df.columns:
        df["high_med_burden"] = (df["num_medications"] > 15).astype(int)
        df["med_count_log"] = np.log1p(df["num_medications"])

    # Diagnosis complexity
    if "number_diagnoses" in df.columns:
        df["high_diagnosis_count"] = (df["number_diagnoses"] >= 8).astype(int)

    # A1C and glucose test signals (None = test not done, which is meaningful)
    if "A1Cresult" in df.columns:
        df["A1C_not_tested"] = (df["A1Cresult"] == "None").astype(int)
        df["A1C_abnormal"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
    if "max_glu_serum" in df.columns:
        df["glu_not_tested"] = (df["max_glu_serum"] == "None").astype(int)
        df["glu_abnormal"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)

    # High-risk discharge dispositions
    if "discharge_disposition_id" in df.columns:
        high_risk_dd = {3, 5, 6, 22, 28}  # SNF, other institutions, home health, rehab, psych
        df["high_risk_discharge"] = df["discharge_disposition_id"].isin(high_risk_dd).astype(int)

    # Convert admission/discharge/source IDs to strings for categorical encoding
    for id_col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str)

    # Fill remaining NaN in race
    if "race" in df.columns:
        df["race"] = df["race"].fillna("Unknown")

    # Create binary readmission target
    if create_target and "readmitted" in df.columns:
        df["readmitted_binary"] = (df["readmitted"] != "NO").astype(int)

    return df


def get_encounter_feature_columns(df):
    """Get the feature columns available in the cleaned encounter data."""
    # Numeric features present in df
    num_feats = [c for c in NUMERIC_FEATURES if c in df.columns]

    # Categorical features present in df
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # Medication columns (already ordinal-encoded as int)
    med_feats = [c for c in MED_COLUMNS if c in df.columns]

    return num_feats, cat_feats, med_feats


def create_encounter_preprocessor(df):
    """Create a sklearn ColumnTransformer for encounter data.

    Returns:
        (preprocessor, feature_cols) where feature_cols is the ordered
        list of columns that the preprocessor expects.
    """
    num_feats, cat_feats, med_feats = get_encounter_feature_columns(df)

    transformers = []

    if num_feats:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_feats,
        ))

    if cat_feats:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_feats,
        ))

    if med_feats:
        transformers.append((
            "med",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]),
            med_feats,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    feature_cols = num_feats + cat_feats + med_feats
    return preprocessor, feature_cols


def prepare_encounter_data(filepath=None, target="readmitted_binary",
                           test_size=0.2, random_state=42):
    """Full pipeline: load, clean, split, preprocess encounter data.

    Uses patient-level splitting to avoid data leakage from repeated patients.

    Returns:
        X_train, X_val, y_train, y_val, preprocessor, feature_cols, df_clean
    """
    df = load_raw_encounters(filepath)
    df = clean_encounters(df, filter_deceased=True, create_target=True)

    # Stratified split
    df_train, df_val = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df[target],
    )

    # Drop ID columns
    id_cols = ["encounter_id", "patient_nbr", "readmitted"]
    y_train = df_train[target].values
    y_val = df_val[target].values

    df_train_feat = df_train.drop(
        columns=[c for c in id_cols + [target] if c in df_train.columns],
        errors="ignore",
    )
    df_val_feat = df_val.drop(
        columns=[c for c in id_cols + [target] if c in df_val.columns],
        errors="ignore",
    )

    preprocessor, feature_cols = create_encounter_preprocessor(df_train_feat)

    X_train = preprocessor.fit_transform(df_train_feat[feature_cols])
    X_val = preprocessor.transform(df_val_feat[feature_cols])

    return X_train, X_val, y_train, y_val, preprocessor, feature_cols, df


def preprocess_encounters_for_prediction(df, preprocessor, feature_cols):
    """Apply saved preprocessing to raw encounter data for prediction."""
    df = clean_encounters(df, filter_deceased=False, create_target=False)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0 if col in NUMERIC_FEATURES else "Unknown"

    X = preprocessor.transform(df[feature_cols])
    return X, df


# ═══════════════════════════════════════════════════════════
# NLP / Medication Review Data
# ═══════════════════════════════════════════════════════════

def load_raw_reviews(filepath=None):
    """Load raw medication feedback CSV."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "healthcare_csvs" / "patient_medication_feedback.csv"
    return pd.read_csv(filepath)


def clean_review_text(text):
    """Clean a single review text string."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)       # HTML tags
    text = re.sub(r"http\S+", " ", text)        # URLs
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)  # Special chars
    text = re.sub(r"\s+", " ", text).strip()    # Multiple spaces
    return text


def prepare_review_data(filepath=None, test_size=0.2, random_state=42):
    """Load, clean, and split medication review data.

    Returns:
        texts_train, texts_val, y_train, y_val, df_clean
    """
    df = load_raw_reviews(filepath)

    # Combine text fields
    for col in ["benefitsReview", "sideEffectsReview", "commentsReview"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["review_text"] = (
        df["benefitsReview"].astype(str) + " " +
        df["sideEffectsReview"].astype(str) + " " +
        df["commentsReview"].astype(str)
    ).str.strip()

    # Clean text
    df["review_text_clean"] = df["review_text"].apply(clean_review_text)

    # Drop rows with empty text
    df = df[df["review_text_clean"].str.len() > 10].copy()

    # Map effectiveness to 3 classes
    df["effectiveness_3class"] = df["effectiveness"].map(EFFECTIVENESS_MAP)
    df = df.dropna(subset=["effectiveness_3class"])

    texts = df["review_text_clean"].values
    y = df["effectiveness_3class"].values

    texts_train, texts_val, y_train, y_val = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    return texts_train, texts_val, y_train, y_val, df


def preprocess_reviews_for_prediction(df):
    """Clean raw review data for prediction."""
    df = df.copy()
    for col in ["benefitsReview", "sideEffectsReview", "commentsReview"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["review_text"] = (
        df["benefitsReview"].astype(str) + " " +
        df["sideEffectsReview"].astype(str) + " " +
        df["commentsReview"].astype(str)
    ).str.strip()
    df["review_text_clean"] = df["review_text"].apply(clean_review_text)
    return df


# ═══════════════════════════════════════════════════════════
# Retinal Image Data
# ═══════════════════════════════════════════════════════════

def load_retinal_labels(labels_path=None):
    """Load retinal scan labels."""
    if labels_path is None:
        labels_path = RAW_DATA_DIR / "healthcare_csvs" / "retinal_labels.csv"
    return pd.read_csv(labels_path)


def prepare_retinal_data(labels_path=None, images_dir=None,
                         test_size=0.2, random_state=42):
    """Load retinal labels, match to images, create binary target.

    Returns:
        train_df, val_df (each with columns: image_path, label)
    """
    if images_dir is None:
        images_dir = RAW_DATA_DIR / "retinal_scan_images"
    images_dir = Path(images_dir)

    labels = load_retinal_labels(labels_path)

    # Match labels to existing images
    existing_images = set(p.stem for p in images_dir.glob("*.png"))
    labels = labels[labels["id_code"].isin(existing_images)].copy()

    # Binary: 0 = No DR, 1 = Has DR
    labels["binary_label"] = (labels["diagnosis"] > 0).astype(int)
    labels["image_path"] = labels["id_code"].apply(
        lambda x: str(images_dir / f"{x}.png")
    )

    train_df, val_df = train_test_split(
        labels, test_size=test_size, random_state=random_state,
        stratify=labels["binary_label"],
    )
    return train_df, val_df


# ═══════════════════════════════════════════════════════════
# Length of Stay (Model 5) Data
# ═══════════════════════════════════════════════════════════

LOS_BINS = {
    "short_stay": (1, 3),   # 1-3 days
    "medium_stay": (4, 7),  # 4-7 days
    "long_stay": (8, 100),  # 8+ days
}


def categorize_los(days):
    """Categorize length of stay as binary: short vs extended."""
    if days <= 4:
        return "short_stay"
    else:
        return "extended_stay"


def prepare_los_data(filepath=None, test_size=0.2, random_state=42):
    """Prepare data for length-of-stay prediction.

    Uses encounter data but with time_in_hospital as target.

    Returns:
        X_train, X_val, y_train, y_val, preprocessor, feature_cols, df_clean
    """
    df = load_raw_encounters(filepath)
    df = clean_encounters(df, filter_deceased=False, create_target=False)

    # Create LOS target
    df["los_category"] = df["time_in_hospital"].apply(categorize_los)

    # Drop columns not available at admission time or that leak the target
    drop_cols = [
        "encounter_id", "patient_nbr", "readmitted",
        "time_in_hospital",  # This IS the target — don't use as feature
    ]
    df_features = df.drop(
        columns=[c for c in drop_cols if c in df.columns], errors="ignore"
    )

    y = df["los_category"].values

    # Remove time_in_hospital from numeric features for LOS model
    los_numeric = [c for c in NUMERIC_FEATURES if c in df_features.columns
                   and c != "time_in_hospital"]
    los_cat = [c for c in CATEGORICAL_FEATURES if c in df_features.columns]
    los_med = [c for c in MED_COLUMNS if c in df_features.columns]

    feature_cols = los_numeric + los_cat + los_med

    transformers = []
    if los_numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            los_numeric,
        ))
    if los_cat:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            los_cat,
        ))
    if los_med:
        transformers.append((
            "med",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]),
            los_med,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        df_features[feature_cols], y,
        test_size=test_size, random_state=random_state, stratify=y,
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    return X_train, X_val, y_train, y_val, preprocessor, feature_cols, df


# ═══════════════════════════════════════════════════════════
# Utility: Find test data files
# ═══════════════════════════════════════════════════════════

def find_test_csv(test_dir, expected_columns=None, name_hint=None):
    """Find the appropriate test CSV in the test_data directory."""
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_dir}")

    csvs = list(test_dir.glob("*.csv"))

    # Try name hint first
    if name_hint:
        for csv_path in csvs:
            if name_hint.lower() in csv_path.name.lower():
                return csv_path

    # Try matching expected columns
    if expected_columns:
        for csv_path in csvs:
            try:
                cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
                if all(c in cols for c in expected_columns):
                    return csv_path
            except Exception:
                continue

    # Fallback: return first CSV that isn't a results file
    for csv_path in csvs:
        if "results" not in csv_path.name.lower():
            return csv_path

    raise FileNotFoundError(
        f"No suitable test CSV found in {test_dir}. Files: {[p.name for p in csvs]}"
    )


def find_test_images(test_dir):
    """Find test images directory."""
    test_dir = Path(test_dir)
    # Check common locations
    for subdir in ["retinal_scan_images", "images", "retinal_images", ""]:
        candidate = test_dir / subdir if subdir else test_dir
        pngs = list(candidate.glob("*.png"))
        if pngs:
            return candidate
    raise FileNotFoundError(f"No PNG images found in {test_dir}")


# ═══════════════════════════════════════════════════════════
# Legacy compat
# ═══════════════════════════════════════════════════════════

def load_raw_data(filename):
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        # Try under healthcare_csvs subdirectory
        filepath = RAW_DATA_DIR / "healthcare_csvs" / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_csv(filepath)


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )


def save_processed_data(df, filename):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_processed_data(filename):
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found: {filepath}")
    return pd.read_csv(filepath)
