import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
st.set_page_config(
    page_title="NO2 Tahmin Sistemi",
    page_icon="🌫️",
    layout="wide"
)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
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
@st.cache_resource
def load_model():
    scaler = joblib.load("scaler.save")
    input_size = scaler.n_features_in_
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
    return model, scaler
model, scaler = load_model()
st.title("🌫️ NO2 Hava Kalitesi Tahmin Sistemi")
st.write("LSTM modeli ile sonraki saatin NO2 seviyesini tahmin edin.")
st.divider()
st.subheader("72 Saatlik Veri Girişi")
st.write(
    "Her satır şu sırada 7 özellik içermelidir:"
)
st.code(
    "NO2(GT), T, RH, AH, CO(GT), hour, month"
)
default_data = np.zeros((72, 7))
default_data[:, 5] = np.arange(72) % 24
default_data[:, 6] = 1
sequence_text = st.text_area(
    "72 satırlık veriyi girin",
    value="\n".join(
        ",".join(
            str(round(value, 4))
            for value in row
        )
        for row in default_data
    ),
    height=400
)
if st.button("NO2 Tahmini Yap", type="primary"):
    try:
        rows = sequence_text.strip().split("\n")
        if len(rows) != 72:
            st.error(
                f"Tam olarak 72 satır girmelisiniz. Şu anda {len(rows)} satır var."
            )
            st.stop()
        data = []
        for row in rows:
            values = [
                float(x.strip())
                for x in row.split(",")
            ]
            if len(values) != 7:
                st.error(
                    "Her satır tam olarak 7 özellik içermelidir."
                )
                st.stop()
            data.append(values)
        data = np.array(
            data,
            dtype=np.float32
        )
        scaled_data = scaler.transform(data)
        X = scaled_data.reshape(
            1,
            72,
            7
        )
        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )
        with torch.no_grad():
            prediction = model(X_tensor).numpy()[0][0]
        dummy = np.zeros(
            (1, scaler.n_features_in_)
        )
        dummy[0, 0] = prediction
        final_prediction = scaler.inverse_transform(
            dummy
        )[0, 0]
        st.divider()
        st.subheader("Tahmin Sonucu")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Tahmin Edilen NO2",
                f"{final_prediction:.2f}"
            )
        with col2:
            st.metric(
                "Kullanılan Zaman Adımı",
                "72 saat"
            )
        with col3:
            st.metric(
                "Model",
                "LSTM"
            )
        st.success(
            f"Sonraki saat için tahmin edilen NO2(GT) değeri: {final_prediction:.2f}"
        )
        st.subheader("Girilen NO2 Değerleri")
        st.line_chart(
            data[:, 0]
        )
    except ValueError:
        st.error(
            "Veri formatında hata var. Sadece sayısal değerler girin."
        )
    except Exception as e:
        st.error(
            f"Bir hata oluştu: {str(e)}"
        )