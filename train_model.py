import pandas as pd

import sklearn

print(sklearn.__version__)
print(sklearn.__file__)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv("headlines_for_labeling.csv")

# ensure no missing labels/text
df = df.dropna(subset=["headline", "label"])

X = df["headline"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9,
        lowercase=True
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions, digits=3))
print("\nLabel distribution:")
print(df["label"].value_counts())