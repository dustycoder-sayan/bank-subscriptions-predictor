# Bank Marketing - Term Deposit Subscription Prediction

## a. Problem Statement

Banks running telephone marketing campaigns contact a large number of clients to
offer a term deposit product, but only a small fraction actually subscribe. Calling
every client in the database is expensive and inefficient, and calling the wrong
clients wastes both staff time and marketing budget. This project builds and
compares five classification models that predict whether a client will subscribe
to a term deposit (`yes`/`no`) based on their demographic, financial, and campaign
contact information - so a bank can prioritize outreach toward clients most likely
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

- **Target variable:** `y` - whether the client subscribed to a term deposit
  (`yes` / `no`)
- **Training samples:** 4,069
- **Test samples:** 454
- **Class imbalance:** The target is heavily skewed toward `no` (most clients
  decline) - `88.5%`, which is why this project reports Precision, Recall, F1, AUC, and MCC
  alongside Accuracy - a model that always predicts `no` would still score high
  on accuracy alone while being useless in practice.
- **`duration` feature:** Each model was trained in two variants - one *including*
  the last call's duration, and one *excluding* it. `duration` is only known
  *after* a call has happened, so a model trained with it isn't usable for
  deciding who to call in advance; it's kept as a secondary model which can be used
  once `duration` of a call is known. 

### Primary Metric

#### False Positives vs. False Negatives - Business Impact

In this context, a **positive prediction** means "this client will subscribe to a term deposit."

**False Positive (predicted "yes", actual "no")**
The bank spends time and money contacting a client who was never going to subscribe. This costs **staff hours, call costs, and minor opportunity cost** - but it's a bounded, low-severity loss. The campaign simply spent effort on someone who said no.

**False Negative (predicted "no", actual "yes")**
The bank **skips a client who would have subscribed**, because the model didn't flag them as a good target. This is a **missed revenue opportunity** - a real customer and real deposit that the bank never got a chance to pursue at all. Unlike a false positive, this loss isn't just wasted effort; it's a lost conversion that's unlikely to be recovered.

**Which is more costly?**
Given how skewed this dataset is (~88% "no", ~12% "yes"), successful conversions are already rare and valuable. **False negatives are the more expensive error** - the cost of a wasted phone call is small and recoverable, while the cost of missing a genuine subscriber directly costs the bank real revenue. This asymmetry is why the model should be tuned to avoid missing "yes" clients, even if that means tolerating more false positives - A reason for the `threshold` used to predict from predict_proba in each model, overriding the default 0.5 threshold to reduce the number of False Negatives. 

---

#### Why Recall Over Accuracy or Precision

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- **Accuracy** is misleading here: since ~88% of clients are "no," a model that predicts "no" for everyone scores ~88% accuracy while catching **zero** actual subscribers - the metric hides the failure that matters most to the business.
- **Precision** measures how many of the predicted "yes" clients actually subscribe - useful for controlling wasted calls, but it says nothing about how many real subscribers were missed entirely.
- **Recall** directly measures what matters most here: **of all the clients who would have said yes, how many did the model actually catch?** Since false negatives (missed subscribers) are the costlier error, recall - which is explicitly the inverse of the false negative rate - is the metric most aligned with the bank's actual business objective: not letting potential subscribers slip through uncontacted.

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

- **Logistic Regression** - a linear model estimating class probability; fast,
  interpretable, and a strong baseline.
- **Decision Tree** - splits data on feature thresholds into a series of
  if-else rules; easy to interpret, prone to overfitting without tuning.
- **K-Nearest Neighbors (kNN)** - classifies a point by majority vote among its
  closest neighbors in feature space; sensitive to feature scaling.
- **Gaussian Naive Bayes** - a probabilistic classification algorithm based on Bayes' theorem, 
  assuming that continuous features associated with each class follow a normal (Gaussian) distribution
  assumptions.
- **Random Forest (Ensemble)** - an ensemble of decision trees trained on
  bootstrapped samples; generally more robust than a single tree.

Each model was trained on two variants of the same dataset -
- With the `duration` feature : Model set to be used when the duration of a 
  call is known.
- Without the `duration` feature : Model set to be used when the duration of a 
  call is `NOT` known.

When using in the application, based on the dataset provided, the appropriate
model is automatically chosen. 

### Comparison Table - With Duration Feature

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression |0.866|0.879|0.450|0.723|0.555|0.500|
| Decision Tree |0.843|0.624|0.326|0.340|0.333|0.244|
| kNN |0.822|0.744|0.345|0.606|0.440154|0.363|
| Gaussian Naive Bayes |0.720|0.800|0.252|0.723|0.374|0.301|
| Random Forest (Ensemble) |0.838|0.890|0.401|0.819|0.538|0.496|

### Comparison Table - Without Duration Feature

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression |0.738|0.704|0.223|0.511|0.311|0.202|
| Decision Tree |0.818|0.583|0.245|0.277|0.260|0.157|
| kNN |0.728|0.600|0.181|0.383|0.246|0.116|
| Gaussian Naive Bayes |0.714|0.695|0.209|0.532|0.300|0.189|
| Random Forest (Ensemble) |0.808|0.692|0.299|0.489|0.371|0.277|

---

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression |The strongest all-round performer on the without-duration set - highest AUC (0.704) and second-best recall (0.511), showing that even a simple linear boundary captures most of the real signal in this data. With duration added, it becomes the single best model by MCC (0.500) and F1 (0.555), and its recall jumps to 0.723, confirming duration is a genuinely powerful (if impractical-to-use-in-advance) predictor.|
| Decision Tree |	The weakest model in both variants despite having the highest accuracy without duration (0.818) - a textbook case of accuracy being misleading under class imbalance. Its recall (0.277) and AUC (0.583) are the worst of the five, meaning it's mostly just predicting the majority class ("no") and getting rewarded for it by the accuracy metric alone. An untuned, unconstrained tree overfitting to noise is the likely cause.|
| kNN |Consistently the second-weakest model. Without duration it has the lowest AUC (0.600) and weakest precision (0.181) of all five, suggesting the raw feature space doesn't separate cleanly by distance - likely hurt by the one-hot encoded categorical features diluting the influence of the more informative numeric ones. Duration helps it more than most (recall rises from 0.383 to 0.606), since it's a single dominant numeric feature that plays to KNN's distance-based strength.|
| Naive Bayes |The clear recall specialist - highest recall in both variants (0.532 without duration, tied-highest 0.723 with duration) - but at a real precision cost (0.209 / 0.252, the lowest of all five both times). This is the expected trade-off from its independence assumption: it casts a wide net and rarely misses a true "yes," but calls "yes" too often to be precise. Best choice if minimizing missed subscribers is the only priority; weakest if wasted calls need controlling.|
| Random Forest (Ensemble) |The most balanced model overall - best MCC (0.277 / 0.496) and best F1 (0.371 / 0.538) in both variants, plus the best precision without duration (0.299) and the best recall and AUC with duration (0.819 / 0.890). It's the only model that performs strongly across every metric simultaneously, rather than trading one off hard against another like GNB (recall vs. precision) or the Decision Tree (accuracy vs. everything else).|
| **Overall Winner for your dataset?** |**Random Forest (Ensemble)** - on the without-duration set (the realistic, deployable model, since duration isn't known before a call is made), it has the best MCC and F1 of all five, the best precision, and solidly mid-to-high recall - no other model is this strong across the board without a major weakness elsewhere. Logistic Regression is the closest runner-up (best AUC, close recall), and is a reasonable second choice if interpretability matters more than squeezing out the last bit of performance.|

---

### Additional Observation on the feature `duration`

- Despite being known only after a call ends, `duration` is the strongest numerical predictor for the dataset. 
- Thus, it can be used when a call has ended and a company wants to then and there keep a track of whether or not the customer is interested to subscribe to the ***Term Deposit*** Subscription provided by the bank.
- However, if the bank does not have the `duration` field yet and wants to know whether a call is to succeed (a more real-world scenario), the models without the duration feature enabled can be used.
- The deployed application automatically detects the variant of model needed and uses the same - thus providing for flexibility in use-cases.

---

## Running the App

To use the features, checkout the deployed application on Streamlit Cloud -\
[https://bank-subscriptions-predictor-2025ac05252-sayanbasu.streamlit.app/]([https://bank-subscriptions-predictor-sayanbasu.streamlit.app/](https://bank-subscriptions-predictor-2025ac05252-sayanbasu.streamlit.app/))

To clone and run it on your local machine - 
```bash
pip install -r requirements.txt
pip install streamlit   # If you want to run the app locally as well
streamlit run app.py
```

---
