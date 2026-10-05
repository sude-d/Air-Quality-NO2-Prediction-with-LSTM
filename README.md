# Air Quality NO2 Prediction with LSTM

## English

An LSTM-based time series forecasting project that predicts the next-hour NO2 concentration using the Air Quality UCI dataset.

### Features

- 72-hour sliding window
- LSTM-based time series prediction
- MinMaxScaler preprocessing
- PyTorch model training
- FastAPI prediction API
- Streamlit web interface
- Model evaluation with MAE, MSE and RMSE

### Technologies

Python, PyTorch, Pandas, NumPy, Scikit-learn, FastAPI, Streamlit

### Project Structure

- `diliupro_preprocessing.py` - Data preprocessing
- `diliupro_train.py` - LSTM model training
- `diliupro_test.py` - Model evaluation
- `diliupro_api.py` - FastAPI service
- `diliupro_app_streamlit.py` - Streamlit interface
- `lstm_air_quality_model.pth` - Trained model
- `scaler.save` - Saved scaler

### Run Streamlit

```bash
python -m streamlit run diliupro_app_streamlit.py
