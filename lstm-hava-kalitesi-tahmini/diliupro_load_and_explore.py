#LSTM ile hava kalitesi tahmini(NO2)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("AirQualityUCI.csv",sep=";",decimal=",",encoding="latin1")
df.dropna(axis=1,how="all",inplace=True)
df["datetime"]=pd.to_datetime(df["Date"]+" "+df["Time"],format="%d/%m/%Y %H.%M.%S",errors="coerce")
df.dropna(subset=["datetime"],inplace=True)
df.drop(["Date","Time"],axis=1,inplace=True)
df.set_index("datetime",inplace=True)
df.replace(-200,np.nan,inplace=True)
df.interpolate(method="time",inplace=True)
df["hour"]=df.index.hour
df["month"]=df.index.month
selected_colums=["NO2(GT)","T","RH","AH","CO(GT)","hour","month"]
df=df[selected_colums]
plt.figure()
sns.heatmap(df.corr(),annot=True,fmt=".2f",cmap="YlGnBu")
plt.title("Korelasyon Matrisi")
plt.show()
plt.figure()
df["NO2(GT)"].plot()
plt.title("NO2(GT) Zaman Serisi")
plt.xlabel("Tarih")
plt.ylabel("NO(GT)")
plt.show()
