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
- **Training samples:** 4,068
- **Test samples:** 453
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
| Logistic Regression |0.821|0.867|0.359|0.711|0.477|0.416|
| Decision Tree |0.859|0.644|0.380|0.365|0.372|0.293|
| kNN |0.812|0.695|0.306|0.500|0.380|0.288|
| Gaussian Naive Bayes |0.786|0.766|0.281|0.558|0.374|0.284|
| Random Forest (Ensemble) |0.839|0.833|0.394|0.750|0.516|0.463|

### Comparison Table - Without Duration Feature

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression |0.801|0.753|0.284|0.481|0.357|0.261|
| Decision Tree |0.814|0.586|0.242|0.288|0.263|0.159|
| kNN |0.737|0.592|0.181|0.365|0.242|0.114|
| Gaussian Naive Bayes |0.773|0.710|0.261|0.538|0.352|0.256|
| Random Forest (Ensemble) |0.744|0.753|0.242|0.577|0.341|0.245|

---

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression |The most balanced performer without duration — highest precision (0.284), F1 (0.357), and MCC (0.261) of all five, and tied for the best AUC (0.753) alongside Random Forest. With duration added it stays strong (MCC 0.416, recall 0.712), confirming the linear boundary captures most of the real signal in this data even without the dominant duration feature.|
| Decision Tree |	A case study in accuracy being misleading: highest accuracy in both variants (0.815 without duration, 0.859 with) but the worst AUC in both (0.586 / 0.644) and the worst or near-worst recall (0.288 / 0.365). It's defaulting heavily toward predicting "no" and getting rewarded for it by accuracy alone — the metric this project deliberately avoids leaning on for exactly this reason.|
| kNN |	The weakest model in both variants — lowest AUC (0.592 without duration, 0.695 with) and lowest precision, recall, F1, and MCC without duration. The raw feature space likely doesn't separate cleanly by distance once categorical one-hot columns dilute the informative numeric ones; it improves somewhat with duration added (recall 0.365 → 0.500) since that's a single strong numeric signal that suits distance-based methods better.|
| Gaussian Naive Bayes |A consistent, solid second-place finisher — second-best F1 (0.352) and MCC (0.256) without duration, trailing Logistic Regression by only a hair. It doesn't lead any single metric outright this time, but it's never the weakest either, making it a dependable, low-variance choice across both variants.|
| Random Forest (Ensemble) |The clear recall leader — best recall in both variants (0.577 without duration, 0.750 with) and tied-best AUC without duration (0.753), plus the best AUC, precision, F1, and MCC of all five with duration included. Its precision/MCC lag slightly behind Logistic Regression on the without-duration set, so it trades a little balance for meaningfully higher recall — catching more true subscribers at the cost of a few more false positives.|
| **Overall Winner for your dataset?** |**Random Forest (Ensemble)** - on the without-duration set, it ties for the best AUC and has clearly the best recall (0.577 vs. 0.481 for the runner-up), which matters most given this project's own framing: missing a real subscriber (false negative) is costlier than one extra wasted call (false positive). Logistic Regression is the strongest alternative if a single balanced score (MCC 0.261 vs. 0.245) is prioritized over recall specifically — the two are close enough that either is defensible depending on which trade-off the bank cares about more.|

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
