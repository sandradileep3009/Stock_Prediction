import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle

df = pd.read_csv("price.csv")
print("Original Shape:", df.shape)
print(df.head())

#CLEAN DATA
df["DATE"] = pd.to_datetime(df["DATE"],errors="coerce")
numeric_cols = [
    "OPEN",
    "HIGH",
    "LOW",
    "PREV. CLOSE",
    "LTP",
    "CLOSE",
    "VWAP",
    "52W H",
    "52W L",
    "VOLUME",
    "VALUE",
    "NO. OF  TRADES"
]

for col in numeric_cols:
    df[col] = (df[col].astype(str).str.replace(",", "", regex=False).str.strip())
    df[col] = pd.to_numeric(df[col],errors="coerce")

df = df.dropna(subset=["CLOSE"])
df = df.sort_values("DATE")


df["RETURN"] = ( df["CLOSE"].pct_change())
df["MA5"] = (df["CLOSE"].rolling(5).mean())
df["MA20"] = (df["CLOSE"].rolling(20).mean())
df["VOLATILITY"] = (df["RETURN"].rolling(20).std())
df=df.dropna()


#VISUALIZATION
plt.figure(figsize=(12,6))
plt.plot(df["DATE"],df["CLOSE"],label="Close")
plt.plot(df["DATE"],df["MA5"],label="MA5")
plt.plot(df["DATE"],df["MA20"],label="MA20")
plt.title("Moving Average")
plt.legend()
plt.show()

features = ["CLOSE","VOLUME","MA5","MA20","RETURN","HIGH","LOW"]
data=df[features]
X=[]
y=[]
window=5
for i in range(len(data)-window):
    X.append(data.iloc[i:i+window].values.flatten())
    y.append(data.iloc[i+window]["CLOSE"])

X=np.array(X)
y=np.array(y)
print("\nML Shapes")
print(X.shape)
print(y.shape)

split=int(len(X)*0.8)
x_train=X[:split]
x_test=X[split:]
y_train=y[:split]
y_test=y[split:]


lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
lr_pred=lr_model.predict(x_test)
lr_mae=mean_absolute_error(y_test,lr_pred)
lr_r2=r2_score(y_test,lr_pred)

print("\nLINEAR REGRESSION")
print("MAE:",lr_mae)
print("R2:",lr_r2)


#SAVE RESULTS
results=pd.DataFrame({"Actual":y_test,"Predicted":lr_pred})
results.to_csv("predictions.csv",index=False)
print("Saved predictions.csv")

# PLOT
plt.figure(figsize=(12,6))
plt.plot(y_test,label="Actual")
plt.plot(lr_pred,label="Prediction")
plt.title("Linear Regression Prediction")
plt.legend()
plt.show()


#LSTM MODEL
lstm_features=[
"CLOSE",
"VOLUME",
"MA5",
"MA20",
"RETURN",
"HIGH",
"LOW"
]
lstm_data=df[lstm_features]
scaler=MinMaxScaler()
scaled=scaler.fit_transform(lstm_data)
X_lstm=[]
y_lstm=[]
window=60
for i in range(window,len(scaled)):
    X_lstm.append(scaled[i-window:i])
    y_lstm.append(scaled[i,0])

X_lstm=np.array(X_lstm)
y_lstm=np.array(y_lstm)
print("\nLSTM Shape")
print(X_lstm.shape)
print(y_lstm.shape)


split=int(len(X_lstm)*0.8)
X_train=X_lstm[:split]
X_test=X_lstm[split:]
y_train=y_lstm[:split]
y_test=y_lstm[split:]


#BUILD LSTM
model_lstm=Sequential([Input(shape=(window,len(lstm_features))),LSTM(64,return_sequences=True),Dropout(0.2),LSTM(64),Dropout(0.2),Dense(25),Dense(1)])
model_lstm.compile(optimizer="adam",loss="mse")
print("\nTraining LSTM")
history=model_lstm.fit(X_train,y_train,epochs=20,batch_size=16,validation_data=(X_test,y_test))

#LSTM PREDICTION
pred=model_lstm.predict(X_test)
pred=np.array(pred)
dummy=np.zeros((pred.shape[0],len(lstm_features)))
dummy[:,0]=pred[:,0]
pred_real=scaler.inverse_transform(dummy)[:,0]
dummy2=np.zeros((y_test.shape[0],len(lstm_features)))
dummy2[:,0]=y_test
actual_real=scaler.inverse_transform(dummy2)[:,0]
lstm_mae=mean_absolute_error(actual_real,pred_real)
print("\nLSTM MAE:")
print(lstm_mae)


#SAVE LSTM
model_lstm.save("stock_lstm_model.keras")
print("Saved LSTM Model")


#TOMORROW PREDICTION

def predict_next_day():
    last_60_days = scaled[-60:]
    input_data = last_60_days.reshape(1,60,len(lstm_features))
    prediction=model_lstm.predict(input_data)
    dummy=np.zeros((1,len(lstm_features)))
    dummy[0,0]=prediction[0][0]
    final_price=scaler.inverse_transform(dummy)[0,0]
    print("\nTomorrow predicted CLOSE:",final_price)

predict_next_day()

with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)