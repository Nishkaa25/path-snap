"""PathSnap evaluation settings wired to the existing src/ml code."""
from pathlib import Path

# This config lives in <project-root>/member_c_eval/config.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw trip folders must be copied to data/train and data/student_test.
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_PARTITION = "train"
STUDENT_TEST_PARTITION = "student_test"

# Existing src/ml/train.py creates this processed candidate-level training CSV.
TRAIN_CSV = PROJECT_ROOT / "results" / "ml" / "candidate_training_data.csv"
STUDENT_TEST_CSV = PROJECT_ROOT / "results" / "ml" / "student_test_candidate_data.csv"
DATA_SOURCE = "csv"

# Match the real columns created by src/ml/dataset.py.
TRIP_COLUMN = "trip_id"
OBSERVATION_COLUMN = "observation_index"
CANDIDATE_ARC_COLUMN = "arc_index"
TARGET_COLUMN = "label"
TRUE_ARC_COLUMN = "true_arc_id"
TIMESTAMP_COLUMN = "timestamp"
TOTAL_OBSERVATION_COLUMN = "total_observations"

# Do not include identifiers, candidate ID, target or added evaluation metadata
# as model inputs. Reuse the feature list from the team's feature engineering.
from src.ml.features import FEATURE_NAMES  # noqa: E402
FEATURE_COLUMNS = list(FEATURE_NAMES)

# Use the five classifiers already implemented in src/ml/train.py.
CUSTOM_MODEL_EXPORTS = [
    (PROJECT_ROOT / "src" / "ml" / "train.py", "models"),
]
USE_DEFAULT_BASELINES = False

CV_SPLITS = 5
RANDOM_STATE = 42
N_BAYESIAN_TRIALS = 30
TUNING_MODEL_NAME = None  # None selects best mean GroupKFold candidate Macro-F1.

# Parameter keys below match each estimator's get_params() exactly.
CUSTOM_SEARCH_SPACES = {
    "LogisticRegression": {
        "model__C": ("float", 1e-3, 100.0, "log"),
        "model__class_weight": ("categorical", [None, "balanced"]),
    },
    "RandomForest": {
        "n_estimators": ("int", 150, 600),
        "max_depth": ("categorical", [None, 8, 12, 16, 24, 32]),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("categorical", ["sqrt", "log2"]),
        "class_weight": ("categorical", [None, "balanced", "balanced_subsample"]),
    },
    "HistGradientBoosting": {
        "learning_rate": ("float", 0.02, 0.25, "log"),
        "max_iter": ("int", 100, 500),
        "max_leaf_nodes": ("int", 15, 63),
        "l2_regularization": ("float", 1e-6, 10.0, "log"),
    },
    "SVM_RBF": {
        "model__C": ("float", 1e-2, 100.0, "log"),
        "model__gamma": ("float", 1e-4, 1e-1, "log"),
        "model__class_weight": ("categorical", [None, "balanced"]),
    },
    "MLP": {
        "model__alpha": ("float", 1e-5, 1e-1, "log"),
        "model__learning_rate_init": ("float", 1e-4, 1e-2, "log"),
        "model__batch_size": ("categorical", [64, 128, 256]),
    },
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "ml" / "member_c_evaluation"
MAX_CONTIGUOUS_GAP_SECONDS = 5.0
OPTUNA_STUDY_NAME_PREFIX = "pathsnap_member_c_tpe"
OPTUNA_STORAGE_FILENAME = "pathsnap_tuning_study.sqlite3"
