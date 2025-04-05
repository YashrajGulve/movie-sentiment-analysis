✅ 1. Objective
This project is designed to classify movie reviews as positive or negative using Traditional Machine Learning (Logistic Regression). It accepts user input, processes it using a TF-IDF vectorizer, and outputs a prediction.

🧠 2. Technologies & Libraries Used
- Python 3.11+
- Flask – for web framework
- scikit-learn – for model and preprocessing
- pandas – for data handling
- nltk – for text cleaning
- gunicorn – for production server
- Render – for deployment

📁 3. Folder Structure
movie-sentiment-analysis/
├── app.py                 # Main Flask app
├── model.pkl              # Trained ML model
├── tfidf.pkl         # TF-IDF vectorizer
├── requirements.txt       # Project dependencies
├── Procfile               # Deployment config for Render
└── templates/             
    ├── index.html         # Review input form
    └── dashboard.html     # Dashboard with logs & pie chart

🧹 4. Data Preprocessing Steps
1. Dataset: IMDB Dataset.csv
2. Removed null values
3. Cleaned text (lowercased, removed punctuation, stopwords, etc.)
4. Split into train-test sets
5. Vectorized using TF-IDF
6. Trained using Logistic Regression

🤖 5. Model Details
- Model Used: LogisticRegression
- Vectorizer: TfidfVectorizer
- Accuracy: ~88% on test data
- Files Saved:
  - model.pkl – the trained model
  - vectorizer.pkl – the TF-IDF transformer

🌐 6. Application Flow
1. User enters a movie review on index.html
2. Review is cleaned & vectorized
3. Model predicts sentiment (positive/negative)
4. Dashboard logs predictions with a pie chart summary

📊 7. Features
- Clean & responsive UI
- Real-time sentiment prediction
- Dashboard with:
  - Review history
  - Pie chart visualization

🚀 8. Deployment
- Hosted on Render
- Used gunicorn for production
- Deployed from GitHub with:
  - Procfile
  - requirements.txt

📌 9. How to Run Locally
git clone https://github.com/YashrajGulve/movie-sentiment-analysis.git
cd movie-sentiment-analysis
pip install -r requirements.txt
python app.py
