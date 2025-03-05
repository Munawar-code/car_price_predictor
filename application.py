import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

# Load dataset to extract unique values for dropdowns
car = pd.read_csv("Cleaned_Cars (2).csv")

@app.route('/')
def index():
    car_names = car['car_name'].dropna().unique().tolist()
    fuel_types = car['fuel'].dropna().unique().tolist()
    body_types = car['body'].dropna().unique().tolist()
    transmission_types = car['transmission'].dropna().unique().tolist()
    year = sorted(car['year'].unique(), reverse=True)
    mileage = car['mileage'].unique()
    return render_template("index.html", car_names=car_names, fuel_types=fuel_types,
                           body_types=body_types, transmission_types=transmission_types,
                           year=year, mileage=mileage)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        car_name = request.form["car_name"]
        car_body = request.form["body"]
        year_of_production = int(request.form["year"])
        mileage = int(request.form["kilo_driven"])
        transmission = request.form["transmission"]
        fuel_type = request.form["fuel"]

        # Ensure inputs match expected training data
        input_data = pd.DataFrame([[car_name, car_body, year_of_production, mileage, fuel_type, transmission]],
                                  columns=['car_name', 'body', 'year', 'mileage', 'fuel', 'transmission'])

        # Replace unknown categorical values with "other"
        for col in ['car_name', 'body', 'fuel', 'transmission']:
            if input_data[col].values[0] not in car[col].unique():
                input_data[col] = "other"

        # Check for missing values
        if input_data.isnull().values.any():
            return "Error: Some fields contain missing values."

        # Make the prediction
        prediction = model.predict(input_data)

        return f"Predicted Car Price: {prediction[0]:,.2f}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

