from flask import Flask, request, render_template
import pickle
import re
import pandas as pd
import os

app = Flask(__name__)

# Load model, vectorizer, and label encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to clean input text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)
    sentiment = label_encoder.inverse_transform(pred)[0]

    # Save to CSV
    df_new = pd.DataFrame([[review, sentiment]], columns=['review', 'prediction'])
    if os.path.exists('predictions.csv'):
        df_new.to_csv('predictions.csv', mode='a', header=False, index=False)
    else:
        df_new.to_csv('predictions.csv', index=False)

    return render_template('index.html', prediction=sentiment)

# Dashboard route with pie chart data
@app.route('/dashboard')
def dashboard():
    if os.path.exists('predictions.csv'):
        df = pd.read_csv('predictions.csv')
        predictions = df.to_dict(orient='records')

        # Count sentiment categories
        positive_count = (df['prediction'] == 'positive').sum()
        negative_count = (df['prediction'] == 'negative').sum()
    else:
        predictions = []
        positive_count = 0
        negative_count = 0

    return render_template(
        'dashboard.html',
        predictions=predictions,
        positive_count=positive_count,
        negative_count=negative_count
    )

if __name__ == '__main__':
    app.run(debug=True)
