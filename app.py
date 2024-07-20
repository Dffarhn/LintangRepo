from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__, template_folder='templates')

# Load the CSV files globally
data_path = "data/audi.csv"
mobil_data = pd.read_csv(data_path)

data_path2 = "data/history.csv"
history_data = pd.read_csv(data_path2)

# Load the pre-trained model
model = joblib.load('decision_tree_model.sav')  # Adjust the path as necessary

# Create a numerical transformer pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

@app.route('/')
def home():
    return render_template("home.html")

def ValuePredictor(to_predict_list):
    try:
        # Convert the input list to DataFrame
        to_predict = pd.DataFrame([to_predict_list], columns=['year', 'mileage', 'tax', 'mpg', 'engineSize'])
        
        # Apply the same preprocessing
        to_predict = numerical_transformer.fit_transform(to_predict)
        
        # Predict using the model
        result = model.predict(to_predict)
        return result[0]
    except Exception as e:
        print(e)
        return None

@app.route('/', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            to_predict_dict = request.form.to_dict()
            to_predict_list = [
                int(to_predict_dict['year']),
                float(to_predict_dict['mileage']),
                float(to_predict_dict['tax']),
                float(to_predict_dict['mpg']),
                float(to_predict_dict['engineSize'])
            ]
            
            result = ValuePredictor(to_predict_list)
            
            if result is not None:
                # Get form data and convert to appropriate types
                new_record = {
                    'year': int(to_predict_dict['year']),
                    'mileage': float(to_predict_dict['mileage']),
                    'tax': float(to_predict_dict['tax']),
                    'mpg': float(to_predict_dict['mpg']),
                    'engineSize': float(to_predict_dict['engineSize']),
                    'price': float(result)
                }
                
                # Create a DataFrame for the new record
                new_record_df = pd.DataFrame([new_record])
                
                # Update history_data globally and save to history.csv
                global history_data
                history_data = pd.concat([history_data, new_record_df], ignore_index=True)
                
                # Save back to CSV
                history_data.to_csv(data_path2, index=False)
                return render_template("home.html", result=f"Price: £ {result}")
            else:
                return render_template("home.html", result="Error in prediction")
        except Exception as e:
            print(e)
            return "Terjadi kesalahan dalam prediksi"

@app.route('/list')
def data_list():
    rows = mobil_data.tail(10).to_dict(orient='records')  # Get last 10 rows as a list of dictionaries
    return render_template("list.html", rows=rows)

@app.route('/history')
def data_history():
    rows = history_data.tail(10).to_dict(orient='records')  # Get last 10 rows as a list of dictionaries
    return render_template("history.html", rows=rows)

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            # Get form data and convert to appropriate types
            new_record = {
                'year': int(request.form['year']),
                'mileage': float(request.form['mileage']),
                'tax': float(request.form['tax']),
                'mpg': float(request.form['mpg']),
                'engineSize': float(request.form['engineSize']),
                'price': float(request.form['price'])
            }
            
            # Create a DataFrame for the new record
            new_record_df = pd.DataFrame([new_record])
            
            # Append new record to the global DataFrame
            global mobil_data
            mobil_data = pd.concat([mobil_data, new_record_df], ignore_index=True)
            
            # Save back to CSV
            mobil_data.to_csv(data_path, index=False)
            
            return redirect(url_for('data_list'))
        except Exception as e:
            print(e)
            return "Error in adding record"
    return render_template("addrec.html")

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3002)
