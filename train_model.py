import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys
import time

from feature_extraction import extract_features, FEATURE_COLUMNS

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"


def main():
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

    urls = df['URL'].tolist()

    # --- Feature extraction ------------------------------------------------
    print(f"Extracting features from {len(urls):,} URLs...")
    t0 = time.time()

    # Simple feature extraction without batch processing
    features_list = [extract_features(url) for url in urls]

    print(f"Feature extraction done in {time.time() - t0:.1f}s")

    # Lock column order to FEATURE_COLUMNS so training always matches what
    # app.py builds at inference time, regardless of dict ordering.
    X = pd.DataFrame(features_list, columns=FEATURE_COLUMNS)
    y = df['label'].astype(int)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Total features: {len(FEATURE_COLUMNS)} (15 original + 5 critical advanced)")
    print(f"New advanced features: full_url_entropy, full_nan_entropy, domain_entropy, digit_ratio, special_char_ratio")
    
    print(f"\nPhishing URLs  : {(y == 1).sum():,}")
    print(f"Legitimate URLs: {(y == 0).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTraining on {len(X_train):,} samples...")

    # n_jobs=-1 already parallelizes tree-building across cores; this part
    # was fine before and is unchanged.
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
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


if __name__ == "__main__":
    # Main execution guard - good practice for all Python scripts
    main()