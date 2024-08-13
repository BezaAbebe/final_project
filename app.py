import os
import io
from flask import Flask, request, render_template, send_from_directory
import pickle
import pandas as pd
import boto3
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
load_dotenv()



# Create an app object using the Flask class
app = Flask(__name__, template_folder='template')


# Function to load the model from S3
def load_model_from_s3(bucket_name, model_file_path, model_name = "random_forest_model.pkl"):
    # Set up AWS credentials
    aws_access_key_id = os.getenv("aws_access_key_id")
    aws_secret_access_key = os.getenv("aws_secret_access_key")
    aws_default_region = os.getenv("aws_default_region")

    if os.path.exists(model_name):
        print("Model exists already, loading model")
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
    else:
        print("Model doesn't exist already, downloading model")
        # # Create an S3 client
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key, region_name=aws_default_region)

        # Download the model from S3 and save it locally
        with open(model_name, 'wb') as f:
            s3.download_fileobj(bucket_name, model_file_path, f)
        
        # Load the model from the downloaded file
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
    print(type(model))
    return model

#  # Load the model once when the app starts
model = load_model_from_s3(bucket_name = 'us-accidents-final', 
                           model_file_path = 'models/random_forest_model.pkl')


# Serve static files from HTML_JS/assets
@app.route('/assets/<path:filename>')
def custom_static(filename):
    return send_from_directory('HTML_JS/assets', filename)

@app.route('/html_js/<path:filename>')
def custom_html_js(filename):
    return send_from_directory('HTML_JS', filename)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    current_dir = os.getcwd()
    return render_template("index.html")

@app.route('/model_predict')
def machinelearning():
    return render_template('model_predict.html')

@app.route('/predict', methods=['POST'])
def predict(model):
  if request.method == 'POST':
        # Collect input data from the form
        state = request.form['state']
        temperature = float(request.form['temperature'])
        wind_speed = float(request.form['wind_speed'])
        precipitation = int(request.form['precipitation'])
        sunrise_sunset = request.form['sunrise_sunset']
        weather_bin = int(request.form['weather_bin'])
        hour_bin = int(request.form['hour_bin'])
        time_duration_bin = int(request.form['time_duration_bin'])
        distance_bin = int(request.form['distance_bin'])
        season = int(request.form['season'])

        # Prepare input data for the model
        input_variables = pd.DataFrame([[state, temperature, wind_speed, precipitation, 
                                         sunrise_sunset, weather_bin, hour_bin, 
                                         time_duration_bin, distance_bin, season]],
                                       columns=['State', 'Temperature', 'Wind_Speed', 
                                                'Precipitation', 'Sunrise_Sunset', 
                                                'Weather_Bin', 'Hour_Bin', 
                                                'Time_Duration_Bin', 'Distance_Bin', 
                                                'Season'],
                                       dtype=float)
        
        # Make prediction using the model
        prediction = model.predict(input_variables)[0]

        # Render the result on the HTML page
        return render_template('model_predict.html', prediction=prediction)
    

if __name__ == '__main__':
    app.run(debug=True)