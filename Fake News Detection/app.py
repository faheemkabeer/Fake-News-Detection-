from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key!

# Dummy credentials for demonstration (replace this with a proper user database later)
USERNAME = 'Faheem'
PASSWORD = '1234'

# Load models
xgb_model = joblib.load("models/xgboost_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
tokenizer = BertTokenizer.from_pretrained("models/bert_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            return redirect(url_for("index"))
        else:
            error = "Invalid Credentials. Try again."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        news_text = request.form.get("news")
        if news_text:
            result = combined_prediction(news_text)
    return render_template("index.html", result=result)

def predict_news_xgboost(news_text):
    prediction = xgb_model.predict(vectorizer.transform([news_text]))[0]
    return prediction

def predict_news_bert(news_text):
    encoding = tokenizer(news_text, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        prediction = torch.argmax(bert_model(**encoding).logits, dim=1).item()
    return prediction

def combined_prediction(news_text):
    xgb_result = predict_news_xgboost(news_text)
    bert_result = predict_news_bert(news_text)
    if xgb_result == bert_result:
        return "Real News" if xgb_result == 1 else "Fake News"
    else:
        return "Uncertain (Disagreement between models)"

if __name__ == "__main__":
    app.run(debug=True)