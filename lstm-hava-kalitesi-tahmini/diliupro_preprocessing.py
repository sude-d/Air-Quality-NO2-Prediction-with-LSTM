import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
df = pd.read_csv(
    "AirQualityUCI.csv",
    sep=";",
    decimal=",",
    encoding="latin1"
)
df.drop(
    columns=df.columns[df.isna().all()],
    inplace=True
)
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H.%M.%S",
    errors="coerce"
)
df.dropna(
    subset=["datetime"],
    inplace=True
)
df.drop(
    ["Date", "Time"],
    axis=1,
    inplace=True
)
df.set_index(
    "datetime",
    inplace=True
)
df.replace(
    -200,
    np.nan,
    inplace=True
)
df.interpolate(
    method="time",
    inplace=True
)
df["hour"] = df.index.hour
df["month"] = df.index.month
selected_columns = [
    "NO2(GT)",
    "T",
    "RH",
    "AH",
    "CO(GT)",
    "hour",
    "month"
]
df = df[selected_columns]
df.dropna(
    inplace=True
)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(
    df
)
joblib.dump(
    scaler,
    "scaler.save"
)
def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        seq_x = data[i:i + window_size]
        seq_y = data[i + window_size][0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
WINDOW_SIZE = 72
X, y = create_sequences(
    scaled,
    WINDOW_SIZE
)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
np.save(
    "X_train.npy",
    X_train
)
np.save(
    "X_test.npy",
    X_test
)
np.save(
    "y_train.npy",
    y_train
)
np.save(
    "y_test.npy",
    y_test
)
print("Veri hazırlama tamamlandı.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("Scaler kaydedildi: scaler.save")