# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 22:10:28 2025

@author: Mauro
"""

# --- CI-friendly plotting (no GUI) ---
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- Paths ---
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(filename: str, dpi: int = 200):
    """Save current matplotlib figure into results/ and close it (prevents memory buildup in CI)."""
    out_path = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")

# -----------------------------
# Stellar Data
# -----------------------------
# Load the data - parquet file
df_stellar = pd.read_parquet(DATA_DIR / "kic_selected_features.parquet", engine="fastparquet")

# Data info
print(df_stellar.head())
print(df_stellar.shape)

# -----------------------------
# KOI data
# -----------------------------
koi_df = pd.read_csv(DATA_DIR / "koi_cumulative_data.csv")

# Data info
print(koi_df.head())
print(koi_df.info())

# -----------------------------
# Data Preparation
# -----------------------------
print(df_stellar.columns)  # checking KIds are the same in both df
print(koi_df.columns)

# Rename the keplerID mismatching column
df_stellar.rename(columns={"kepler_id": "kepid"}, inplace=True)

# Filtering
filtered_df_stellar = df_stellar[df_stellar["kepid"].isin(koi_df["kepid"])]

# Check result
print(filtered_df_stellar.shape)
print(filtered_df_stellar.head())

# Duplicates in both datasets
duplicates_koi = koi_df[koi_df.duplicated(subset="kepid", keep=False)]
print(f"Number of duplicated kepid in koi_df: {duplicates_koi.shape[0]}")
print(duplicates_koi.head())

duplicates_stellar = filtered_df_stellar[filtered_df_stellar.duplicated(subset="kepid", keep=False)]
print(f"Number of duplicated kepid in df_stellar: {duplicates_stellar.shape[0]}")
print(duplicates_stellar.head())

# -----------------------------
# Merging the datasets
# -----------------------------
df_merged = koi_df.merge(df_stellar, on="kepid", how="inner")
print(df_merged.info())

# Missing values
missing = df_merged.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Dropping unnecessary features with missing data
drop_cols_missing = [
    "koi_teq_err1", "koi_teq_err2",
    "kic_parallax", "kepler_name",
    "koi_score", "koi_tce_delivname"
]
df_merged.drop(columns=drop_cols_missing, inplace=True)

# Checking distribution for missing values features
def validate_imputation(feature):
    plt.figure(figsize=(12, 5))

    # Plot with missing values
    plt.subplot(1, 2, 1)
    sns.histplot(df_merged[feature], kde=True, bins=30)
    plt.title(f"{feature} (with missing)")

    # Plot without missing values
    plt.subplot(1, 2, 2)
    sns.histplot(df_merged[feature].dropna(), kde=True, bins=30)
    plt.axvline(df_merged[feature].median(), linestyle="--", label="Median")
    plt.title(f"{feature} (non-missing only)")
    plt.legend()

    save_fig(f"imputation_check_{feature}.png")

    # Print basic stats
    missing_ratio = df_merged[feature].isna().mean()
    print(f"Missing ratio for {feature}: {missing_ratio:.2%}")
    print(f"Mean: {df_merged[feature].mean():.2f}")
    print(f"Median: {df_merged[feature].median():.2f}")
    print("=" * 50)

for col in ["kic_teff", "koi_prad", "koi_depth"]:
    validate_imputation(col)

# Filling missing with median
df_merged = df_merged.fillna(df_merged.median(numeric_only=True))

print("Remaining missing values:", df_merged.isnull().sum().sum())

# -----------------------------
# Low variance features
# -----------------------------
numeric_df = df_merged.select_dtypes(include=["number"])

selector = VarianceThreshold(threshold=0.0)
selector.fit(numeric_df)
retained_columns = numeric_df.columns[selector.get_support()]
df_high_variance = numeric_df[retained_columns]

dropped = set(numeric_df.columns) - set(retained_columns)
print(f"Dropped low-variance features: {dropped}")

variances = df_merged.select_dtypes(include=["number"]).var().sort_values(ascending=True)
print(variances.head(30))

std_devs = df_merged.select_dtypes(include=["number"]).std().sort_values()
cv = (df_merged.select_dtypes(include=["number"]).std() /
      df_merged.select_dtypes(include=["number"]).mean()).sort_values()

print("Lowest Coefficient of Variation (CV):")
print(cv.head(30))

drop_low_value_features = [
    "koi_period_err1", "koi_period_err2",
    "kic_pmra", "kic_pmdec",
    "koi_time0bk_err1", "koi_time0bk_err2",
    "koi_slogg_err1", "koi_slogg_err2",
    "koi_prad_err2", "koi_depth_err2",
    "koi_insol_err2", "koi_srad_err2",
    "koi_impact_err2", "koi_duration_err2",
    "koi_steff_err2", "kic_blend", "kic_variable"
]
df_merged.drop(columns=drop_low_value_features, inplace=True)

variances = df_merged.select_dtypes(include=["number"]).var().sort_values(ascending=True)
print(variances.head(10))

std_devs = df_merged.select_dtypes(include=["number"]).std().sort_values()
cv = (df_merged.select_dtypes(include=["number"]).std() /
      df_merged.select_dtypes(include=["number"]).mean()).sort_values()

print("Lowest Coefficient of Variation (CV):")
print(cv.head(10))

# -----------------------------
# Correlation
# -----------------------------
corr = df_merged.select_dtypes(include="number").corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features")
save_fig("correlation_matrix.png")

corr_pairs = corr.abs().unstack().sort_values(ascending=False)
high_corr = corr_pairs[(corr_pairs < 1.0) & (corr_pairs > 0.9)]
print(high_corr)

drop_corr_features = [
    "kic_kepmag",
    "kepid",
    "koi_insol_err1",
    "kic_teff"
]
df_merged.drop(columns=drop_corr_features, inplace=True)

# -----------------------------
# Outliers
# -----------------------------
feature_ranges = df_merged.describe().loc[["min", "max"]].T
feature_ranges["range"] = feature_ranges["max"] - feature_ranges["min"]
print("Top 20 features with highest range:")
print(feature_ranges.sort_values(by="range", ascending=False).head(20))

def detect_outliers_all(df, exclude=None, threshold=1.5):
    if exclude is None:
        exclude = []
    outlier_counts = {}
    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_counts[col] = len(outliers)

    return pd.Series(outlier_counts).sort_values(ascending=False)

binary_cols = ["koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_nt", "koi_fpflag_ec"]
outlier_summary = detect_outliers_all(df_merged, exclude=binary_cols + ["label"])
print("Features with detected outliers:")
print(outlier_summary[outlier_summary > 0])

# Copy the dataframe
df_treated = df_merged.copy()

# log1p transformation
log_transform_features = [
    "koi_depth", "koi_prad", "koi_period", "koi_model_snr",
    "koi_prad_err1", "koi_depth_err1", "koi_insol", "koi_teq"
]

for col in log_transform_features:
    df_treated[col] = np.log1p(df_treated[col])
    print(f"Log1p applied to: {col}")

# Winsorize features
winsorize_features = [
    "koi_duration", "koi_impact", "koi_srad", "kic_radius", "koi_time0bk",
    "koi_duration_err1", "koi_impact_err1", "koi_srad_err1",
    "koi_steff", "koi_steff_err1", "kic_logg", "koi_slogg", "kic_feh"
]

for col in winsorize_features:
    series = df_treated[col]
    df_treated[col] = winsorize(series, limits=[0.01, 0.01])
    print(f"Winsorized: {col}")

feature_ranges = df_treated.describe().loc[["min", "max"]].T
feature_ranges["range"] = feature_ranges["max"] - feature_ranges["min"]
scaled_candidates = feature_ranges.sort_values(by="range", ascending=False)
print(scaled_candidates.head(20))

# -----------------------------
# Scaling
# -----------------------------
df_final_scaled = df_treated.copy()
final_scale_features = ["dec", "koi_kepmag"]

scaler_final = StandardScaler()
df_final_scaled[final_scale_features] = scaler_final.fit_transform(df_final_scaled[final_scale_features])
print(df_final_scaled[final_scale_features].agg(["min", "max", "mean", "std"]).T)

# -----------------------------
# EDA
# -----------------------------
label_map = {
    "FALSE POSITIVE": 0,
    "CANDIDATE": 1,
    "CONFIRMED": 2
}

df_final_scaled["target"] = df_final_scaled["koi_disposition"].map(label_map)
print(df_final_scaled["target"].value_counts())
df_final_scaled = df_final_scaled.drop(columns="label", errors="ignore")

X = df_final_scaled.drop(columns=["target", "koi_disposition"])
y = df_final_scaled["target"]

X = X.select_dtypes(include="number")

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 6))
importances.sort_values(ascending=False).head(15).plot(kind="barh", title="Top 15 Feature Importances")
plt.xlabel("Importance Score")
save_fig("top15_feature_importances.png")

# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df_final_scaled, x="target")
plt.title("Distribution of Exoplanet Labels")
plt.xticks([0, 1, 2], ["False Positive", "Candidate", "Confirmed"])
save_fig("target_distribution.png")

# Flag_NT
plt.figure(figsize=(7, 5))
sns.countplot(data=df_final_scaled, x="target", hue="koi_fpflag_nt", palette="Set2")
plt.title("Not Transit-Like Flag by target")
plt.xlabel("target")
plt.ylabel("Count")
plt.legend(title="koi_fpflag_nt")
plt.xticks([0, 1, 2], ["False Positive", "Candidate", "Confirmed"])
save_fig("flag_nt_by_target.png")

# Flag_co
plt.figure(figsize=(7, 5))
sns.countplot(data=df_final_scaled, x="target", hue="koi_fpflag_co")
plt.title("Centroid Offset Flag by target")
plt.xlabel("target")
plt.ylabel("Count")
plt.legend(title="koi_fpflag_co")
plt.xticks([0, 1, 2], ["False Positive", "Candidate", "Confirmed"])
save_fig("flag_co_by_target.png")

# fpflag_ss
plt.figure(figsize=(7, 5))
sns.countplot(data=df_final_scaled, x="target", hue="koi_fpflag_ss")
plt.title("Significant Secondary Flag by target")
plt.xlabel("target")
plt.ylabel("Count")
plt.legend(title="koi_fpflag_ss")
plt.xticks([0, 1, 2], ["False Positive", "Candidate", "Confirmed"])
save_fig("flag_ss_by_target.png")

# model_snr
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_final_scaled, x="target", y="koi_model_snr")
plt.title("Signal-to-Noise Ratio by target")
plt.xlabel("target")
plt.ylabel("koi_model_snr")
plt.xticks([0, 1, 2], ["False Positive", "Candidate", "Confirmed"])
save_fig("box_model_snr_by_target.png")

# koi_prad
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_final_scaled, x="target", y="koi_prad")
plt.title("Planet Radius by target")
plt.xlabel("target")
plt.ylabel("koi_prad")
plt.xticks([0, 1, 2], ["False Positive", "Candidate", "Confirmed"])
save_fig("box_koi_prad_by_target.png")

df_final_scaled.info()

# -----------------------------
# Feature engineering
# -----------------------------
valid_df = df_final_scaled.copy()

valid_df["planet_star_ratio"] = valid_df["koi_prad"] / (valid_df["koi_srad"] + 1e-6)
valid_df["depth_per_prad"] = valid_df["koi_depth"] / (valid_df["koi_prad"] + 1e-6)
valid_df["duration_ratio"] = valid_df["koi_duration"] / (valid_df["koi_period"] + 1e-6)

print(valid_df[["planet_star_ratio", "depth_per_prad", "duration_ratio"]].describe())

# -----------------------------
# Data processing for models
# -----------------------------
df_tree = valid_df.copy()
df_other = valid_df.copy()

X_tree = df_tree.drop(columns=["target", "koi_disposition"]).select_dtypes(include="number")
y_tree = df_tree["target"]

X_other = df_other.drop(columns=["target", "koi_disposition"]).select_dtypes(include="number")
y_other = df_other["target"]

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_tree, y_tree, test_size=0.2, stratify=y_tree, random_state=42
)

X_train_other, X_test_other, y_train_other, y_test_other = train_test_split(
    X_other, y_other, test_size=0.2, stratify=y_other, random_state=42
)

# Scale (distance models)
scaler = StandardScaler()
X_train_other_scaled = scaler.fit_transform(X_train_other)
X_test_other_scaled = scaler.transform(X_test_other)

# SMOTE
smote = SMOTE(random_state=42)
X_train_other_bal, y_train_other_bal = smote.fit_resample(X_train_other_scaled, y_train_other)

print("Original class distribution (non-tree):")
print(y_train_other.value_counts())
print("\nBalanced class distribution (after SMOTE):")
print(pd.Series(y_train_other_bal).value_counts())

# -----------------------------
# Models
# -----------------------------

# RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=1500,
    max_depth=40,
    min_samples_split=20,
    min_samples_leaf=6,
    max_features="sqrt",
    bootstrap=False,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_tree, y_train_tree)

y_pred_rf = rf.predict(X_test_tree)
y_proba_rf = rf.predict_proba(X_test_tree)

print("Random Forest – Tuned Parameters")
print(classification_report(y_test_tree, y_pred_rf, digits=3))

cm = confusion_matrix(y_test_tree, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["False Positive", "Candidate", "Confirmed"],
    yticklabels=["False Positive", "Candidate", "Confirmed"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Random Forest (Tuned)")
save_fig("cm_random_forest.png")

confidences = np.max(y_proba_rf, axis=1)
correctness = (y_pred_rf == y_test_tree).astype(int)

plt.figure(figsize=(9, 5))
plt.hist(confidences[correctness == 1], bins=20, alpha=0.75, label="Correct")
plt.hist(confidences[correctness == 0], bins=20, alpha=0.75, label="Incorrect")
plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Samples")
plt.title("Confidence vs Correctness – Random Forest (Tuned)")
plt.legend()
save_fig("confidence_hist_random_forest.png")

# XGBOOST
xgb = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False
)

xgb.fit(
    X_train_tree,
    y_train_tree,
    eval_set=[(X_test_tree, y_test_tree)],
    verbose=False
)

y_pred_xgb = xgb.predict(X_test_tree)
y_pred_proba = xgb.predict_proba(X_test_tree)

print("XGBoost \n", classification_report(y_test_tree, y_pred_xgb, digits=3))
print(f"Best iteration (round): {xgb.best_iteration}")
print(confusion_matrix(y_test_tree, y_pred_xgb))

cm = confusion_matrix(y_test_tree, y_pred_xgb)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["False Positive", "Candidate", "Confirmed"],
    yticklabels=["False Positive", "Candidate", "Confirmed"]
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - XGBoost")
save_fig("cm_xgboost.png")

confidences = np.max(y_pred_proba, axis=1)
correctness = (y_pred_xgb == y_test_tree).astype(int)

plt.figure(figsize=(8, 5))
plt.hist(confidences[correctness == 1], bins=20, alpha=0.7, label="Correct")
plt.hist(confidences[correctness == 0], bins=20, alpha=0.7, label="Incorrect")
plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Samples")
plt.title("Confidence vs Correctness - XGBoost")
plt.legend()
save_fig("confidence_hist_xgboost.png")

# SVM (RBF)
svm = SVC(
    kernel="rbf",
    C=3.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

svm.fit(X_train_other_bal, y_train_other_bal)

y_pred_svm = svm.predict(X_test_other_scaled)
y_proba_svm = svm.predict_proba(X_test_other_scaled)

print("SVM (RBF) – Tuned Parameters")
print(classification_report(y_test_other, y_pred_svm, digits=3))

cm = confusion_matrix(y_test_other, y_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["False Positive", "Candidate", "Confirmed"],
    yticklabels=["False Positive", "Candidate", "Confirmed"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – SVM (RBF)")
save_fig("cm_svm.png")

confidences = np.max(y_proba_svm, axis=1)
correctness = (y_pred_svm == y_test_other).astype(int)

plt.figure(figsize=(9, 5))
plt.hist(confidences[correctness == 1], bins=20, alpha=0.75, label="Correct")
plt.hist(confidences[correctness == 0], bins=20, alpha=0.75, label="Incorrect")
plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Samples")
plt.title("Confidence vs Correctness – SVM (RBF)")
plt.legend()
save_fig("confidence_hist_svm.png")

# KNN
knn = KNeighborsClassifier(
    n_neighbors=10,
    weights="uniform",
    metric="minkowski",
    p=1
)

knn.fit(X_train_other_bal, y_train_other_bal)

y_pred_knn = knn.predict(X_test_other_scaled)
y_proba_knn = knn.predict_proba(X_test_other_scaled)

print("KNN – Tuned Parameters")
print(classification_report(y_test_other, y_pred_knn, digits=3))

cm = confusion_matrix(y_test_other, y_pred_knn)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["False Positive", "Candidate", "Confirmed"],
    yticklabels=["False Positive", "Candidate", "Confirmed"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – KNN")
save_fig("cm_knn.png")

confidences = np.max(y_proba_knn, axis=1)
correctness = (y_pred_knn == y_test_other).astype(int)

plt.figure(figsize=(9, 5))
plt.hist(confidences[correctness == 1], bins=5, alpha=0.75, label="Correct")
plt.hist(confidences[correctness == 0], bins=5, alpha=0.75, label="Incorrect")
plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Samples")
plt.title("Confidence vs Correctness – KNN")
plt.legend()
save_fig("confidence_hist_knn.png")

# MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="tanh",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate="adaptive",
    learning_rate_init=1e-3,
    max_iter=400,
    early_stopping=True,
    n_iter_no_change=15,
    validation_fraction=0.12,
    random_state=42
)

mlp.fit(X_train_other_bal, y_train_other_bal)

y_pred_mlp = mlp.predict(X_test_other_scaled)
y_proba_mlp = mlp.predict_proba(X_test_other_scaled)

print("MLP (sklearn ANN)\n", classification_report(y_test_other, y_pred_mlp, digits=3))

cm = confusion_matrix(y_test_other, y_pred_mlp)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["False Positive", "Candidate", "Confirmed"],
    yticklabels=["False Positive", "Candidate", "Confirmed"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – MLP")
save_fig("cm_mlp.png")

confidences = np.max(y_proba_mlp, axis=1)
correctness = (y_pred_mlp == y_test_other).astype(int)

bins = np.linspace(0.0, 1.0, 21)
plt.figure(figsize=(9, 5))
plt.hist(confidences[correctness == 1], bins=bins, alpha=0.75, label="Correct")
plt.hist(confidences[correctness == 0], bins=bins, alpha=0.75, label="Incorrect")
plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Samples")
plt.title("Confidence vs Correctness – MLP")
plt.legend()
save_fig("confidence_hist_mlp.png")

print("Done. Plots saved to results/")
