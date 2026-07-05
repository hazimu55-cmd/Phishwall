import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

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
    n_workers = os.cpu_count() or 4

    # --- Feature extraction, parallelized across CPU cores --------------
    # extract_features() is pure Python string/regex work with no shared
    # state, so it's a perfect fit for process-based parallelism. Threads
    # would NOT help here (GIL), which is why the original single list
    # comprehension only ever used one core for ~235k rows. Processes
    # sidestep the GIL and use every core.
    print(f"Extracting features from {len(urls):,} URLs using {n_workers} processes...")
    t0 = time.time()

    # chunksize batches work per worker to cut down on IPC overhead for
    # a function this cheap and this numerous.
    chunksize = max(1, len(urls) // (n_workers * 20))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        features_list = list(executor.map(extract_features, urls, chunksize=chunksize))

    print(f"Feature extraction done in {time.time() - t0:.1f}s")

    # Lock column order to FEATURE_COLUMNS so training always matches what
    # app.py builds at inference time, regardless of dict ordering.
    X = pd.DataFrame(features_list, columns=FEATURE_COLUMNS)
    y = df['label'].astype(int)

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
    # Required guard for ProcessPoolExecutor: without it, each spawned
    # worker process would re-import and re-run this module from the top,
    # causing infinite recursive process spawning on Windows/macOS (spawn
    # start method) and duplicated work everywhere else.
    main()