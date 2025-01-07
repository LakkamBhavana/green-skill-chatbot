import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Example questions and categories
data = [
    ("What is renewable energy?", "energy"),
    ("How do solar panels work?", "solar"),
    ("Tell me about climate change", "environment"),
    ("What is wind energy?", "wind"),
    ("Explain recycling", "recycling")
]

# Split data into questions (X) and categories (y)
X = [text for text, label in data]  # Questions
y = [label for text, label in data]  # Categories

# Create a model pipeline with a vectorizer and classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "green_skill_chatbot_model.pkl")
print("Model saved successfully!")
