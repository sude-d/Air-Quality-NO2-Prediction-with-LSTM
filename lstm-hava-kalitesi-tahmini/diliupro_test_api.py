import requests
import numpy as np
X_test=np.load("X_test.npy")
sample_index=0
sample=X_test[sample_index]
payload={
    "sequence":sample.tolist()
    }
url="http://127.0.0.1:8000/predict"
try:
    response=requests.post(url,json=payload)
    if response.status_code==200:
        result=response.json()
        print(f"tahmin edilen NO2 değeri:{result}")
    else:
        print(f"hata:{response.status_code}")
except requests.exceptions.ConnectionError:
    print("FastAPI başlatın")
    

