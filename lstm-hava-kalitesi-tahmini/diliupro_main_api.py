from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
app = FastAPI(
    title="NO2 Tahmin API",
    description="LSTM modeli kullanarak 72 zaman adımı ile NO2 tahmini yapar."
)
scaler = joblib.load("scaler.save")
input_size = scaler.n_features_in_
hidden_size = 64
num_layers = 2
dropout = 0.2
output_size = 1
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
class SequenceInput(BaseModel):
    sequence: list[list[float]]
@app.get("/")
def home():
    return {
        "message": "NO2 Tahmin API çalışıyor"
    }
@app.post("/predict")
def predict(data: SequenceInput):
    seq = data.sequence
    if len(seq) != 72:
        raise HTTPException(
            status_code=400,
            detail="Sequence tam olarak 72 zaman adımı içermelidir."
        )
    if any(len(row) != input_size for row in seq):
        raise HTTPException(
            status_code=400,
            detail=f"Her zaman adımı {input_size} özellik içermelidir."
        )
    try:
        X = np.array(
            seq,
            dtype=np.float32
        )
        X = X.reshape(
            1,
            72,
            input_size
        )
        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )
        with torch.no_grad():
            prediction = model(
                X_tensor
            ).numpy()[0][0]
        dummy = np.zeros(
            (1, scaler.n_features_in_)
        )
        dummy[0][0] = prediction
        inv_pred = scaler.inverse_transform(
            dummy
        )[0][0]
        return {
            "tahmin_edilen_NO2": round(
                float(inv_pred),
                2
            )
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
