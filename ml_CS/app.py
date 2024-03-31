from flask import Flask, render_template, request
import pandas as pd  # Assuming you have pandas installed
from sklearn.externals import joblib  # Assuming you have scikit-learn installed

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("trained_model.pkl")  # You need to replace "trained_model.pkl" with your actual model file

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict sales price
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    manufacturer = request.form['manufacturer']
    model = request.form['model']
    engine_size = float(request.form['engine_size'])
    fuel_efficiency = float(request.form['fuel_efficiency'])
    vehicle_type = request.form['vehicle_type']

    # Prepare data for prediction
    data = pd.DataFrame({
        'manufacturer': [manufacturer],
        'model': [model],
        'engine_size': [engine_size],
        'fuel_efficiency': [fuel_efficiency],
        'vehicle_type': [vehicle_type]
    })

    # Make prediction
    predicted_price = model.predict(data)[0]

    return render_template('index.html', prediction_text='Predicted Sales Price: ${:.2f}'.format(predicted_price))

if __name__ == '__main__':
    app.run(debug=True)
