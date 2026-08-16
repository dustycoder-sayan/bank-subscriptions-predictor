# Bank Marketing — Term Deposit Subscription Prediction

## a. Problem Statement

Banks running telephone marketing campaigns contact a large number of clients to
offer a term deposit product, but only a small fraction actually subscribe. Calling
every client in the database is expensive and inefficient, and calling the wrong
clients wastes both staff time and marketing budget. This project builds and
compares five classification models that predict whether a client will subscribe
to a term deposit (`yes`/`no`) based on their demographic, financial, and campaign
contact information — so a bank can prioritize outreach toward clients most likely
to convert, rather than contacting everyone.

---

## b. Dataset Description

This project uses the **[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)**,
sourced from direct marketing campaigns (phone calls) run by a Portuguese banking
institution. Each row represents one client contact, described by demographic
attributes (age, job, marital status, education), financial attributes (average
yearly balance, existing housing/personal loans, credit default history), and
campaign contact details (contact type, day/month of last contact, number of
contacts, days since previous contact, outcome of the previous campaign).

- **Target variable:** `y` — whether the client subscribed to a term deposit
  (`yes` / `no`)
- **Training samples:** 4,069
- **Test samples:** 454
- **Class imbalance:** The target is heavily skewed toward `no` (most clients
  decline), which is why this project reports Precision, Recall, F1, AUC, and MCC
  alongside Accuracy — a model that always predicts `no` would still score high
  on accuracy alone while being useless in practice.
- **`duration` feature:** Each model was trained in two variants — one *including*
  the last call's duration, and one *excluding* it. `duration` is only known
  *after* a call has happened, so a model trained with it isn't usable for
  deciding who to call in advance; it's kept as a secondary model which can be used
  once `duration` of a call is known. 

---

## c. GitHub Repository Link

[https://github.com/dustycoder-sayan/bank-subscriptions-predictor](https://github.com/dustycoder-sayan/bank-subscriptions-predictor)

The repository contains all files required to reproduce and run this project:

```
project-folder/
├── app.py                    # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv
├── train_data.csv
└── model/
    ├── bank_subscription_classifiers.ipynb   # training, comparison, threshold tuning
    ├── preprocessing.py
    ├── evaluation_metrics.py
    └── pipelines/                            # Saved sklearn pipelines to be called in app 
        ├── without_duration/                 # Models to train on datasets where duration is available
        └── with_duration/                    # Models to train on datasets where duration is NOT available
```

---

## d. Models Used

Five classification algorithms were trained and compared on identical
train/test splits and identical preprocessing:

- **Logistic Regression** — a linear model estimating class probability; fast,
  interpretable, and a strong baseline.
- **Decision Tree** — splits data on feature thresholds into a series of
  if-else rules; easy to interpret, prone to overfitting without tuning.
- **K-Nearest Neighbors (kNN)** — classifies a point by majority vote among its
  closest neighbors in feature space; sensitive to feature scaling.
- **Naive Bayes** — a probabilistic classifier assuming conditional feature
  independence; fast to train and a solid baseline despite its simplifying
  assumptions.
- **Random Forest (Ensemble)** — an ensemble of decision trees trained on
  bootstrapped samples; generally more robust than a single tree.

Each model was trained on two variants of the same dataset -
- With the `duration` feature : Model set to be used when the duration of a 
  call is known.
- Without the `duration` feature : Model set to be used when the duration of a 
  call is `NOT` known.

When using in the application, based on the dataset provided, the appropriate
model is automatically chosen. 

### Comparison Table — Without Duration Feature

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |

### Comparison Table — With Duration Feature

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |

---

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| kNN | |
| Naive Bayes | |
| Random Forest (Ensemble) | |
| **Overall Winner for your dataset?** | |

---

### Additional Observation on the feature `duration`

---

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

---