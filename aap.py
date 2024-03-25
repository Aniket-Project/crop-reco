
import streamlit as st
import numpy as np
import pickle

# Load the model and scalers
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('standscaler.pkl','rb'))
min_max_scaler = pickle.load(open('minmaxscaler.pkl','rb'))

# Dictionary mapping crop numbers to crop names
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Function to scale features
def scale_features(features):
    scaled_features = min_max_scaler.transform(scaler.transform(features))
    return scaled_features

# Function to predict crop
def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    scaled_features = scale_features([feature_list])
    prediction = model.predict(scaled_features)
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        return crop
    else:
        return "Unknown"

# Streamlit Interface
st.title("Crop Recommendation System")
st.write("Enter the soil and environmental parameters to get crop recommendations.")

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Predict"):
    crop = predict_crop(N, P, K, temp, humidity, ph, rainfall)
    st.write(f"Predicted Crop: {crop}")
