import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("training/dataset.csv")

# =========================
# 1. Text Cleaning Function
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return " ".join(text.split())

# =========================
# 2. FEATURE ENGINEERING (IMPORTANT)
# =========================

df['combined_text'] = (
    df['Ticket Subject'].fillna('') + " " +
    df['Ticket Description'].fillna('') + " " +
    df['Ticket Type'].fillna('') + " " +
    df['Product Purchased'].fillna('') + " " +
    df['Ticket Channel'].fillna('')
)

# Clean
df['combined_text'] = df['combined_text'].apply(clean_text)

# =========================
# 3. Features / Labels
# =========================
X = df['combined_text']
y = df['Ticket Type']   # أو Ticket Priority حسب هدفك

# =========================
# 4. Vectorization (UPGRADED)
# =========================
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    min_df=2
)

X_vec = vectorizer.fit_transform(X)

# =========================
# 5. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. Model (Still Naive Bayes but optimized)
# =========================
model = MultinomialNB(alpha=0.5)

model.fit(X_train, y_train)

# =========================
# 7. Save Artifacts
# =========================
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print(" Production model trained successfully!")