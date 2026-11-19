# Predicting term-deposit subscriptions from bank telemarketing data

Can a bank predict, **before making a call**, which clients will subscribe to a term deposit? This project compares a random forest and a Keras neural network on 41,188 real phone contacts from a Portuguese bank.

Final project of the AI Specialisation at Tokio School.

## Results at a glance

| Test set (threshold 0.5) | Random forest | Neural network |
|---|---|---|
| AUC | **0.8094** | 0.7988 |
| Recall | **0.6437** | 0.6422 |
| Precision | **0.4091** | 0.3744 |
| F1-score | **0.5003** | 0.4730 |
| Accuracy | **0.8551** | 0.8388 |

A model that always answers "no" would reach 0.8873 accuracy and find no subscriptions at all, which is why accuracy is not the metric used to choose.

**In business terms:** the random forest flagged 17.7% of the test contacts, and those contacts held 64% of all subscriptions. Calling only those clients, 41 in every 100 calls end in a subscription, against 11 in every 100 when calling everyone.

The random forest was chosen: it is better on every metric, finds the same subscriptions as the network with 100 fewer false alarms, and shows which variables drive the prediction. Its weak point is more overfitting (AUC 0.8837 on train against 0.8094 on test), but the test AUC is very close to the validation AUC (0.8047).

## The data

[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (UCI Machine Learning Repository, also on [Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)), file `bank-additional-full.csv`: 41,188 contacts from May 2008 to November 2010, with client data, campaign history and five economic indicators. Only 11.27% of contacts ended in a subscription.

## Key decisions

- **Removed `duration` (data leakage).** Call length is strongly linked to the outcome, but it is only known after the call. Keeping it would give the model information the bank does not have when it decides whom to call.
- **Hidden missing values.** There were no nulls: missing values were marked as `"unknown"` and as `999` in `pdays`. "unknown" was imputed with the most frequent category in five columns, but kept as a category in `default`, where it carries information (5.2% subscription rate against 12.9% for "no"). `pdays` became the binary feature `contacted_before`.
- **No test leakage in preprocessing.** Standardisation and PCA were fitted on the training set only, after a stratified 70/15/15 split (`random_state=90`).
- **PCA only on the economic indicators.** Three of them are correlated above 0.9. They were reduced to 3 components (99.3% of the variance), while every other feature stayed interpretable.
- **Class weights instead of resampling** (an error on a subscription counts 7.9 times more). No rows are invented or removed.
- **Tuning with AUC**, which measures how well the model ranks clients and does not depend on the threshold.
  - Random forest: `GridSearchCV`, 18 combinations, stratified 5-fold cross-validation. Best: 200 trees, max depth 12, min 5 samples per leaf.
  - Neural network: random search of 8 out of 54 combinations, with early stopping on validation AUC. Best: 1 hidden layer of 16 neurons, dropout 0.2, learning rate 0.005.

## What drives the prediction

The three economic components account for about 49% of the random forest's feature importance: the economic moment of the call matters more than the client's profile. Next come the contact history (`contacted_before`, `poutcome_success`), age and contact type.

## Limitations

- The data covers a period of economic crisis (2008 to 2010), so the model would need retraining on recent data.
- The split is random, not by date. Real use (train on the past, predict the future) was not tested.
- Precision is 41%: the model is useful for deciding whom to call first, not for knowing for sure who will subscribe.
- PCA removes the direct reading of each economic indicator.

Ideas for future work: a date-based split, a threshold based on call cost and subscription value, gradient boosting, SHAP explanations, and a fairness check by age.

## How to run

```
pip install -r requirements.txt
```

Open `bank_marketing_prediction.ipynb` and run all cells. The dataset is included in this repository. The grid search takes a few minutes.

## Data citation

S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, Elsevier, 62:22-31, June 2014. The dataset is shared by the UCI Machine Learning Repository under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.
