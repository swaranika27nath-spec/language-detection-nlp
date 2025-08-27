# language-detection-nlp
Automated language identification of text samples using classical NLP and machine learning methods.
# language_detector.py
# Simple language detection using scikit-learn and TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data - sentences with their language labels
texts = [
    "Hello, how are you?",         # English
    "What is your name?",          # English
    "Bonjour, comment ça va?",     # French
    "Quel est ton nom?",           # French
    "Hola, ¿cómo estás?",          # Spanish
    "¿Cuál es tu nombre?",         # Spanish
    "Hallo, wie geht's dir?",      # German
    "Wie heißt du?",               # German
    "Ciao, come stai?",            # Italian
    "Come ti chiami?",             # Italian
]
labels = [
    "English",
    "English",
    "French",
    "French",
    "Spanish",
    "Spanish",
    "German",
    "German",
    "Italian",
    "Italian"
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Language detection accuracy: {accuracy:.2f}")

# Optional: Predict language for new sentences
sample_sentences = ["Wie geht es Ihnen?", "Good morning!", "¿Dónde está la estación?"]
sample_vec = vectorizer.transform(sample_sentences)
predictions = model.predict(sample_vec)
for i, sentence in enumerate(sample_sentences):
    print(f"'{sentence}' predicted as: {predictions[i]}")
