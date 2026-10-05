
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)
y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1)
scaler = joblib.load(
    "scaler.save"
)
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout
    ):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(
            hidden_size,
            output_size
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
input_size = X_test.shape[2]
hidden_size = 64
num_layers = 2
dropout = 0.2
output_size = 1
model = LSTMModel(
    input_size,
    hidden_size,
    num_layers,
    output_size,
    dropout
)
model.load_state_dict(
    torch.load(
        "lstm_air_quality_model.pth",
        map_location="cpu"
    )
)
model.eval()
with torch.no_grad():
    predictions = model(X_test)
y_pred = predictions.numpy()
y_true = y_test.numpy()
dummy_pred = np.zeros(
    (len(y_pred), scaler.n_features_in_)
)
dummy_true = np.zeros(
    (len(y_true), scaler.n_features_in_)
)
dummy_pred[:, 0] = y_pred[:, 0]
dummy_true[:, 0] = y_true[:, 0]
inv_y_pred = scaler.inverse_transform(
    dummy_pred
)[:, 0]
inv_y_true = scaler.inverse_transform(
    dummy_true
)[:, 0]
mae = mean_absolute_error(
    inv_y_true,
    inv_y_pred
)
mse = mean_squared_error(
    inv_y_true,
    inv_y_pred
)
rmse = np.sqrt(mse)
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
plt.figure(figsize=(14, 6))
plt.plot(
    inv_y_true,
    label="Gerçek NO2",
    linewidth=2
)
plt.plot(
    inv_y_pred,
    label="Tahmin NO2",
    linewidth=2
)
plt.title(
    "LSTM Hava Kalitesi NO2 Tahmini"
)
plt.xlabel(
    "Test Örneği"
)
plt.ylabel(
    "NO2(GT)"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
