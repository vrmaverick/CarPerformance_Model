from flask import Flask, render_template, request
import pandas as pd  # Assuming you have pandas installed
# from sklearn.externals import joblib # Assuming you have scikit-learn installed
# import joblib
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
# ml_model = joblib.load("./Trained_Model/Car_Performance_Model.pkl")  # You need to replace "trained_model.pkl" with your actual model file
with open("./Trained_Model/Car_Performance_Model.pkl", "rb") as f:
    ml_model = pickle.load(f)

# with open("./Trained_Model/encoder", "rb") as f:
#     encoder = pickle.load(f)


# Extract categorical variables used for one-hot encoding
category = ["Vehicle_type", "Model", "Manufacturer"]

# Instantiate ColumnTransformer
# trans = ColumnTransformer([('one_hot', encoder, category)], remainder='passthrough')

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict sales price
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    Manufacturer = request.form['manufacturer']
    Model = request.form['model']
    Engine_size = float(request.form['engine_size'])
    Fuel_efficiency = float(request.form['fuel_efficiency'])
    Vehicle_type = request.form['vehicle_type']

    # Prepare data for prediction
    # data = pd.DataFrame({
    #     'Manufacturer': [manufacturer],
    #     'Model': [model],
    #     'Engine_size': [engine_size],
    #     'Fuel_efficiency': [fuel_efficiency],
    #     'Vehicle_type': [vehicle_type]
    # })
    input_data = np.array([Manufacturer, Model, Vehicle_type, Engine_size, Fuel_efficiency])
    # input_data = np.transpose(input_data)
    input_data = input_data.reshape(1, -1)  # Reshape to (1, 5)
    print(input_data.shape)  # This will print (1, 5)
    print(input_data.shape)

    # Transform input data using ColumnTransformer
    # input_data_transformed = trans.transform(input_data)
    input_data["Vehicle_type", "Model", "Manufacturer"] = pd.get_dummies(input_data , columns = ["Vehicle_type", "Model", "Manufacturer"])
    # input_data = input_data.flatten()
    # Make prediction
    predicted_price = ml_model.predict(input_data)[0]

    return render_template('index.html', prediction_text='Predicted Sales Price: ${:.2f}'.format(predicted_price))

if __name__ == '__main__':
    app.run(debug=True)
