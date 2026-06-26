"""
Run: python pipeline.py            (uses built-in mappings)
Or:  python pipeline.py --csv Login_Data.csv  (trains on real data)
"""

import argparse, os, pickle, re, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor

OUTPUT = os.path.dirname(os.path.abspath(__file__))

DEVICE_MAP = {"mobile": 0, "desktop": 1, "tablet": 2, "bot": 3, "unknown": 4}

BOT_KEYWORDS = ["bot", "awariosmartbot", "metajobbot", "libwwwperl",
    "mobileiron", "coc coc", "woobot", "crawler_faq", "job roboter",
    "keeper", "bingpreview", "nutch", "curl", "okhttp", "zipppbot"]

TOP_BROWSERS = [
    "Chrome", "Chrome Mobile", "Chrome Mobile WebView", "Firefox",
    "Firefox Mobile", "Safari", "Safari Mobile", "Android", "Edge",
    "MiuiBrowser", "Opera", "Opera Mobile", "Samsung Internet",
    "Facebook", "Instagram", "Yandex Browser", "Maxthon", "UC Browser",
    "Vivaldi", "Brave", "QQbrowser", "Sogou Explorer"
]

def categorize_browser(name):
    if not isinstance(name, str):
        return "Others"
    name_clean = re.search(r"^\D+", name)
    name_clean = name_clean.group() if name_clean else name
    if name_clean in TOP_BROWSERS:
        return name_clean
    if any(kw in name_clean.lower() for kw in BOT_KEYWORDS):
        return "Bot"
    return "Others"

def build_preprocess(df):
    print("Building label encoders...")
    country_enc = LabelEncoder()
    browser_enc = LabelEncoder()

    country_enc.fit(df["Country"].astype(str))
    all_browsers = list(set(df["Browser Category"].unique()) | set(TOP_BROWSERS + ["Bot", "Others"]))
    browser_enc.fit(all_browsers)

    with open(os.path.join(OUTPUT, "encoders.pkl"), "wb") as f:
        pickle.dump({"country": country_enc, "browser": browser_enc, "device": DEVICE_MAP}, f)
    print("Saved encoders.pkl")
    return country_enc, browser_enc

def encode_data(df, country_enc, browser_enc):
    data = df.copy()
    data["Country"] = country_enc.transform(data["Country"].astype(str))
    data["Device Type"] = data["Device Type"].map(DEVICE_MAP).fillna(4).astype(int)
    data["Login Successful"] = data["Login Successful"].astype(int)
    data["Browser Category"] = browser_enc.transform(data["Browser Category"])
    return data

def engineer_features(df):
    """Feature engineering from original login data."""
    data = df.copy()
    data["Browser Category"] = data["Browser Name and Version"].apply(categorize_browser)
    data = data.drop(["Browser Name and Version"], axis=1)

    data["NumSuccessfulLogins"] = data.groupby("User ID")["Login Successful"].transform("sum")
    false_counts = data[data["Login Successful"] == 0].groupby("User ID").size().reset_index(name="NumUnsuccessfulLogins")
    data = data.merge(false_counts, on="User ID", how="left")
    data["NumUnsuccessfulLogins"] = data["NumUnsuccessfulLogins"].fillna(0).astype(int)
    data["NumSuccessfulLogins"] = data["NumSuccessfulLogins"].replace(0, 1)
    data["LoginRatio"] = data["NumUnsuccessfulLogins"] / data["NumSuccessfulLogins"]
    data["LoginRatio"] = data["LoginRatio"].replace([np.inf, -np.inf], 0)
    data = data.drop(["NumSuccessfulLogins", "NumUnsuccessfulLogins"], axis=1)

    data["Total Device Types Per User"] = data.groupby("User ID")["Device Type"].transform("nunique")
    data["Total IP Addresses Per User"] = data.groupby("User ID")["IP Address"].transform("nunique")
    data["Total Countries Per User"] = data.groupby("User ID")["Country"].transform("nunique")
    data["Total Browser Categories Per User"] = data.groupby("User ID")["Browser Category"].transform("nunique")

    data["Login Timestamp"] = pd.to_datetime(data["Login Timestamp"])
    data = data.sort_values(["User ID", "Login Timestamp"])
    data["Time Difference"] = data.groupby("User ID")["Login Timestamp"].diff().dt.total_seconds().fillna(0)

    data["target"] = 0
    data.loc[data["Total Countries Per User"] > 2, "target"] += 1
    data.loc[data["Total Device Types Per User"] > 3, "target"] += 1
    data.loc[data["Total IP Addresses Per User"] > 4, "target"] += 1
    data.loc[data["Total Browser Categories Per User"] > 3, "target"] += 1
    data.loc[(data["Time Difference"] > 0) & (data["Time Difference"] < 5), "target"] += 1
    data.loc[data["Browser Category"] == "Bot", "target"] += 2
    data.loc[data["Device Type"] == "bot", "target"] += 2
    data.loc[data["LoginRatio"] > 10, "target"] += 1

    return data

def get_feature_cols():
    return ["Country", "Device Type", "Login Successful", "LoginRatio",
            "Browser Category", "Total Device Types Per User",
            "Total IP Addresses Per User", "Total Countries Per User",
            "Total Browser Categories Per User", "Time Difference"]

def create_synthetic_data():
    print("No CSV provided. Generating synthetic data for encoders + IsolationForest...")
    np.random.seed(42)
    countries = ["US", "NO", "AU", "GB", "DE", "FR", "IN", "BR", "RU", "JP",
                 "CA", "CN", "KR", "SE", "DK", "FI", "NL", "IT", "ES", "PL"]
    devices = list(DEVICE_MAP.keys())
    browsers = TOP_BROWSERS + ["Bot", "Others"]
    import random
    users = [str(random.randint(10**18, 9*10**18)) for _ in range(2000)]

    rows = []
    for _ in range(10000):
        rows.append({
            "Login Timestamp": f"2020-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d} {np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}",
            "User ID": np.random.choice(users),
            "IP Address": f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}",
            "Country": np.random.choice(countries),
            "Browser Name and Version": np.random.choice(browsers) + f" {np.random.randint(1,120)}.0",
            "Device Type": np.random.choice(devices),
            "Login Successful": np.random.choice([True, False], p=[0.6, 0.4]),
        })
    df = pd.DataFrame(rows)
    df = engineer_features(df)

    country_enc, browser_enc = build_preprocess(df)
    encoded = encode_data(df, country_enc, browser_enc)
    return encoded, country_enc, browser_enc

def load_csv_and_train(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}, Cols: {list(df.columns)}")
    df = engineer_features(df)

    country_enc, browser_enc = build_preprocess(df)
    encoded = encode_data(df, country_enc, browser_enc)
    return encoded, country_enc, browser_enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to Login_Data.csv")
    args = parser.parse_args()

    if args.csv:
        encoded, country_enc, browser_enc = load_csv_and_train(args.csv)
    else:
        encoded, country_enc, browser_enc = create_synthetic_data()

    feat_cols = get_feature_cols()
    X = encoded[feat_cols]
    y = encoded["target"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining XGBoost regressor...")
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(y_test, y_pred)
    print(f"XGBoost - MSE: {mse:.6f}, RMSE: {np.sqrt(mse):.6f}, MAE: {mean_absolute_error(y_test, y_pred):.6f}")

    with open(os.path.join(OUTPUT, "regressor_model.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    print("Saved regressor_model.pkl")

    print("\nTraining Isolation Forest (unsupervised)...")
    ifo = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    ifo.fit(X_train)
    with open(os.path.join(OUTPUT, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(ifo, f)
    print("Saved isolation_forest.pkl")

    print(f"\nAll artifacts saved to {OUTPUT}")
    print("Files: encoders.pkl, regressor_model.pkl, isolation_forest.pkl")

if __name__ == "__main__":
    main()
