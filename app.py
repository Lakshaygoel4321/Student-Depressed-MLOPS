from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and preprocessor
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        # Get form values
        data = {
            "Age": int(request.form["Age"]),
            "Gender": request.form["Gender"],
            "City": request.form["City"],
            "Profession": request.form["Profession"],
            "Academic Pressure": float(request.form["Academic Pressure"]),
            "CGPA": float(request.form["CGPA"]),
            "Study Satisfaction": float(request.form["Study Satisfaction"]),
            "Sleep Duration": request.form["Sleep Duration"],
            "Dietary Habits": request.form["Dietary Habits"],
            "Degree": request.form["Degree"],
            "Have you ever had suicidal thoughts ?": request.form["Have you ever had suicidal thoughts ?"],
            "Work/Study Hours": float(request.form["Work/Study Hours"]),
            "Financial Stress": float(request.form["Financial Stress"]),
            "Family History of Mental Illness": request.form["Family History of Mental Illness"]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Transform and predict
        transformed = preprocessor.transform(input_df)
        pred = model.predict(transformed)[0]
        prediction = "Yes, the student is depressed." if pred == 1 else "No, the student is not depressed."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)