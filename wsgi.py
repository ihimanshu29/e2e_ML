# Save this as wsgi.py in your project root

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
# Import the specific class needed
from mlProject.pipeline.prediction import PredictionPipeline 

# Initialize the pipeline object ONCE globally when the application starts
# This makes subsequent predictions much faster!
PREDICTOR_OBJ = PredictionPipeline()

app = Flask(__name__) # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

# The /train route is removed. Training should be done offline/via CI/CD.
# Do NOT run os.system() in a production web app route.

@app.route('/predict', methods=['POST', 'GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # reading the inputs given by the user (Form data is strings, convert them)
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])
        
            
            data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
            data = np.array(data).reshape(1, 11)
            
            # Use the pre-initialized global object
            predict = PREDICTOR_OBJ.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print(f'Prediction Exception: {e}')
            return render_template('error.html', message='An error occurred during prediction.')

    else:
        # GET request or initial load
        return render_template('index.html')


# Gunicorn will handle the production run, so the __main__ block is no longer needed.
# For local testing, you can add it back:
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)