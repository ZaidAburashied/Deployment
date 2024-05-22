import streamlit as st
import pandas as pd
import joblib

INR_to_JD = 851.48

# Load the trained model
model = joblib.load("model.pkl")

# Define the feature order used during training
fit_order = [
    "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "Year_1998", "Year_1999", "Year_2000", "Year_2001",
    "Year_2002", "Year_2003", "Year_2004", "Year_2005", "Year_2006", "Year_2007", "Year_2008", "Year_2009",
    "Year_2010", "Year_2011", "Year_2012", "Year_2013", "Year_2014", "Year_2015", "Year_2016", "Year_2017",
    "Year_2018", "Year_2019", "Fuel_Type_CNG", "Fuel_Type_Diesel", "Fuel_Type_LPG", "Fuel_Type_Petrol",
    "Transmission_Automatic", "Transmission_Manual", "Owner_Type_First", "Owner_Type_Fourth & Above",
    "Owner_Type_Second", "Owner_Type_Third", "company_Ambassador", "company_Audi", "company_BMW", "company_Bentley",
    "company_Chevrolet", "company_Datsun", "company_Fiat", "company_Force", "company_Ford", "company_Honda",
    "company_Hyundai", "company_ISUZU", "company_Isuzu", "company_Jaguar", "company_Jeep", "company_Lamborghini",
    "company_Land", "company_Mahindra", "company_Maruti", "company_Mercedes-Benz", "company_Mini",
    "company_Mitsubishi", "company_Nissan", "company_Porsche", "company_Renault", "company_Skoda", "company_Tata",
    "company_Toyota", "company_Volkswagen", "company_Volvo",
]

# Define function to preprocess user input
def preprocess_input(user_input):
    # Convert user input to DataFrame
    input_df = pd.DataFrame.from_dict([user_input])

    # Define the desired feature order and names
    desired_features = [
        "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", 
        "Year_1998", "Year_1999", "Year_2000", "Year_2001", "Year_2002", 
        "Year_2003", "Year_2004", "Year_2005", "Year_2006", "Year_2007", 
        "Year_2008", "Year_2009", "Year_2010", "Year_2011", "Year_2012", 
        "Year_2013", "Year_2014", "Year_2015", "Year_2016", "Year_2017", 
        "Year_2018", "Year_2019", "Transmission_Automatic", "Transmission_Manual",
        "Fuel_Type_CNG", "Fuel_Type_Diesel", "Fuel_Type_LPG", "Fuel_Type_Petrol",
        "Owner_Type_First", "Owner_Type_Fourth & Above", "Owner_Type_Second", 
        "Owner_Type_Third", "company_Ambassador", "company_Audi", "company_BMW", 
        "company_Bentley", "company_Chevrolet", "company_Datsun", "company_Fiat", 
        "company_Force", "company_Ford", "company_Honda", "company_Hyundai", 
        "company_ISUZU", "company_Isuzu", "company_Jaguar", "company_Jeep", 
        "company_Lamborghini", "company_Land", "company_Mahindra", "company_Maruti", 
        "company_Mercedes-Benz", "company_Mini", "company_Mitsubishi", "company_Nissan", 
        "company_Porsche", "company_Renault", "company_Skoda", "company_Tata", 
        "company_Toyota", "company_Volkswagen", "company_Volvo"
    ]

    # Perform one-hot encoding and reindex to match the desired feature order and names
    input_df_encoded = pd.get_dummies(input_df).reindex(columns=desired_features, fill_value=0)

    return input_df_encoded


# Define function to predict price
def predict_price(inputs):
    # Preprocess input
    input_data = preprocess_input(inputs)

    # Predict price
    predicted_price = model.predict(input_data)
    
    predicted_price_JD = predicted_price[0] * INR_to_JD


    return predicted_price_JD


# Streamlit UI
st.title("Car Price Prediction")

# Input fields
inputs = {}
inputs["Kilometers_Driven"] = st.number_input("Kilometers Driven",step=1000,value=5000)
inputs["Mileage"] = st.number_input("Mileage")
inputs["Engine"] = st.number_input("Engine")
inputs["Power"] = st.number_input("Power")
inputs["Seats"] = st.number_input("Seats", value=2, min_value=2, step=1)
inputs["Year"] = st.number_input("Year", value=2005, min_value=1998, max_value=2019, step=1)
inputs["Fuel_Type"] = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
inputs["Transmission"] = st.selectbox("Transmission", ["Manual", "Automatic"])
inputs["Owner_Type"] = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
company_options = ['Ambassador', 'Audi', 'BMW', 'Bentley', 'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda',
                   'Hyundai', 'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini', 'Land', 'Mahindra', 'Maruti',
                   'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota',
                   'Volkswagen', 'Volvo']

inputs['company'] = st.selectbox("Company", company_options)

# Predict price on button click
if st.button("Predict Price"):
    predicted_price = predict_price(inputs)
    st.success(f"Predicted Price: {predicted_price:.2f} JD")
