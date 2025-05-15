import streamlit as st
import numpy as np
import joblib

st.title("Air Quality Prediction App")

pm25 = st.slider("PM2.5", 0, 500)
pm10 = st.slider("PM10", 0, 500)
no2 = st.slider("NO2", 0, 200)
so2 = st.slider("SO2", 0, 200)
co = st.slider("CO", 0.0, 10.0)
o3 = st.slider("O3", 0, 200)
temp = st.slider("Temperature", -10, 50)
humidity = st.slider("Humidity", 0, 100)

features = np.array([[pm25, pm10, no2, so2, co, o3, temp, humidity]])
model = joblib.load('models/rf_model.pkl')
prediction = model.predict(features)

st.subheader("Predicted AQI:")
st.success(round(prediction[0], 2))
