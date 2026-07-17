Emotion Detector

A text-based emotion classifier built with TF-IDF and Logistic Regression, deployed as an interactive Streamlit web app. Type any sentence and the app predicts which of six emotions it expresses, along with a confidence breakdown across all classes.

Built by Jaswant Gopala.

Features


Predicts one of six emotions from free text: sadness, anger, love, surprise, fear, joy
Custom "emotion spectrum" visualization showing confidence across all six classes, not just the top prediction
Clean, custom-styled UI (not default Streamlit theme)
Lightweight and fast — runs entirely on a classical ML pipeline, no deep learning dependencies


Tech Stack


Python
Scikit-learn — TF-IDF vectorizer + Logistic Regression classifier
Streamlit — web app framework
Pandas — data handling for the probability table


Model Performance

Trained on the Emotions dataset (16,000 labeled sentences across 6 classes), evaluated on a 20% held-out test split.

The training data is imbalanced (joy: 5,362 examples vs. love: 1,304 and surprise: 572), which caused the initial model to under-predict the minority classes. This was fixed by training with class_weight='balanced':

MetricBaselineBalanced (shipped model)Overall accuracy86.3%88.5%Love recall61.5%91.2%Surprise recall46.9%85.8%

A Naive Bayes baseline (Bag-of-Words and TF-IDF) was also tested during development; Logistic Regression + TF-IDF with balanced class weights performed best overall.

How It Works


User enters a sentence in the text box
The text is lowercased and cleaned (URLs and punctuation stripped)
The cleaned text is transformed into TF-IDF features using the saved vectorizer
The trained Logistic Regression model predicts the emotion class and its probability distribution
Results are displayed as a labeled prediction plus a visual spectrum of all six emotion confidences


Project Structure

emotions classifier/
├── app.py                    # Streamlit app
├── requirements.txt          # Python dependencies
├── tfidf_vectorizer.pkl      # Trained TF-IDF vectorizer
├── logistic_model.pkl        # Trained Logistic Regression model
├── emotion_numbers.pkl       # Emotion label ↔ number mapping
└── README.md

Setup & Installation


Clone the repository:


bash   git clone https://github.com/jaswantgopala/Emotion-Detector.git
   cd Emotion-Detector


Install dependencies:


bash   pip install -r requirements.txt


Run the app:


bash   python -m streamlit run app.py


Open the URL shown in the terminal (typically http://localhost:8501) in your browser.


Known Limitations


The model was trained on short, informal text (tweet-style sentences), so longer or more formal sentences may reduce prediction confidence.
The training data has a class imbalance (joy and sadness are overrepresented relative to love and surprise). This was mitigated using class_weight='balanced' during training, which substantially improved recall on the minority classes without hurting overall accuracy.


Future Improvements


Add a confusion matrix visualization to the README/repo for full transparency on per-class performance
Experiment with more expressive models (e.g. transformer-based embeddings) for comparison against the TF-IDF baseline
Deploy publicly via Streamlit Community Cloud


License

This project is open for personal and educational use.
