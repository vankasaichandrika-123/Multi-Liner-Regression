#from flask import Flask, render_template, request

from flask import Flask, render_template, request
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
#import pickle import sklearn import sklearn.linear_model import LinearRegression

# Load trained model
with open('MLR_PKL', 'rb') as f:
    model = pickle.load(f)

# Create Flask App
app = Flask(__name__)

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        # Get input values from HTML form
        features = [float(x) for x in request.form.values()]

        # Convert into numpy array
        final_features = np.array([features])

        # Prediction
        prediction = model.predict(final_features)[0]

        return render_template(
            'index.html',
            prediction_text=f"Predicted Value: {prediction:.2f}"
        )

    return render_template('index.html')


# Run Application
if __name__ == '__main__':
    app.run(debug=True)