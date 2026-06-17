import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# LOAD AND PREPARE DATA


df = pd.read_csv("headlines_for_labeling.csv")
df = df.dropna(subset=["headline", "label"])

X = df["headline"]
y = df["label"]


# Creates the same model architecture for both tasks
def create_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),  # use single words and word pairs
            min_df=2,            # ignore very rare words
            max_df=0.9,
            lowercase=True
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])



# MODEL 1: CLICKBAIT DETECTOR
# 0 = normal news
# 1 = emotional/clickbait (labels 1 + 2)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert labels: 0 stays 0, 1 and 2 become clickbait
y_train_m1 = (y_train != 0).astype(int)
y_test_m1 = (y_test != 0).astype(int)

model1 = create_model()
model1.fit(X_train, y_train_m1)

print("\nMODEL 1 (CLICKBAIT DETECTOR)")
print(classification_report(
    y_test_m1,
    model1.predict(X_test),
    digits=3
))



# MODEL 2: CLICKBAIT TYPE
# 1 = emotional news
# 2 = strong clickbait


clickbait = df[df["label"] != 0]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    clickbait["headline"],
    clickbait["label"],
    test_size=0.2,
    random_state=42,
    stratify=clickbait["label"]
)

model2 = create_model()
model2.fit(X2_train, y2_train)

print("\nMODEL 2 (CLICKBAIT TYPE)")
print(classification_report(
    y2_test,
    model2.predict(X2_test),
    digits=3
))

print("\nLabel distribution:")
print(df["label"].value_counts())



# HEADLINE ANALYSIS


def analyze_headline(text):

    # Model 1 probability:
    # index 0 = normal, index 1 = clickbait
    p_normal, p_clickbait = model1.predict_proba([text])[0]

    # Model 2 probability:
    # returns probabilities for classes 1 and 2
    probs = dict(zip(
        model2.classes_,
        model2.predict_proba([text])[0]
    ))

    # Combine both model decisions:
    # overall probability = clickbait chance * clickbait type chance
    p1 = p_clickbait * probs.get(1, 0)
    p2 = p_clickbait * probs.get(2, 0)

    final = int(np.argmax([p_normal, p1, p2]))

    return p_normal, p1, p2, final



# TEST WITH USER HEADLINES


while True:
    headline = input("\nEnter headline (or 'exit'): ")

    if headline.lower() == "exit":
        break

    p0, p1, p2, result = analyze_headline(headline)

    print("\nAnalysis")
    print("Headline:", headline)
    print(f"P(non-clickbait=0): {p0:.3f}")
    print(f"P(emotional news=1): {p1:.3f}")
    print(f"P(clickbait=2): {p2:.3f}")
    print("Final class:", result)