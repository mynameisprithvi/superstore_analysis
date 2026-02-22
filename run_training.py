import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "superstore_processed.parquet"
MODEL_OUTPUT = BASE_DIR / "models" / "discount_model.joblib"

df = pd.read_parquet(DATA_PATH)

assert "Discount" in df.columns, "Target variable missing"

target = "Discount"
X = df.drop(columns=[target])
y = df[target]

# -----------------------
# Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Preprocessing
# -----------------------
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# -----------------------
# Model
# -----------------------
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# -----------------------
# Train
# -----------------------
pipeline.fit(X_train, y_train)

# -----------------------
# Save model
# -----------------------
MODEL_OUTPUT.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_OUTPUT)

print("Model training complete. Saved to:", MODEL_OUTPUT)