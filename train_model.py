import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys


DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset not found at '{DATASET_PATH}'")
    print("Please download the dataset and place it in the same folder.")
    sys.exit(1)

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH).dropna()
print(f"Loaded {df.shape[0]:,} rows")


if 'URL' not in df.columns or 'label' not in df.columns:
    print("Columns found:", df.columns.tolist())
    print("ERROR: Expected 'URL' and 'label' columns.")
    sys.exit(1)

from feature_extraction import extract_features

print("Extracting features from URLs (this may take a minute)...")
features_list = [extract_features(url) for url in df['URL']]
X = pd.DataFrame(features_list)
y = df['label'].astype(int)

print(f"\nPhishing URLs  : {(y == 1).sum():,}")
print(f"Legitimate URLs: {(y == 0).sum():,}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining on {len(X_train):,} samples...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))


os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/phishing_model.pkl")
print("Model saved to models/phishing_model.pkl")
print("\nNow run: streamlit run app.py")