# Latvian Clickbait Detector

NLP project that classifies Latvian news headlines into three categories:

- 0 — neutral headline
- 1 — emotional news / clickbait-like headline
- 2 — strong clickbait

The project uses a two-stage text classification approach:
1. Detects whether a headline is normal or clickbait-like.
2. Classifies the clickbait intensity.

## Technologies

- Python 3.11.0
- scikit-learn
- pandas
- NumPy

## Methods

- TF-IDF text representation
- Logistic Regression classification
- Hierarchical classification approach


## Tutorial

- Download python
- Make sure you have pip installed
- Run command to download required programs: "pip install pandas numpy scikit-learn"
- Launch command "python train_model.py"