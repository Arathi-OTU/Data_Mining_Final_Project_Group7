from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load saved files
model = joblib.load("sentiment_model_nb.pkl")
vectorizer = joblib.load("tfidf.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    prob = model.predict_proba(review_vector)

    confidence = max(prob[0]) * 100

    if prediction == "positive":
        sentiment = "Positive 😊"
    elif prediction == "negative":
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral"

    return render_template(
        "index.html",
        prediction=sentiment,
        review=review,
        confidence=round(confidence,2)
    )

if __name__ == "__main__":
    app.run(debug=True)