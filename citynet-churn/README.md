# CityNet churn baseline

## User story
As the retention manager, I want a weekly list of the 100 customers most likely to cancel next month, so my team can call them first.

## How to run
    pip install -r requirements.txt
    python src/lab7_save.py                       # train and save the pipeline
    python src/predict.py data/new_customers.csv  # writes reports/predictions.csv

## Results (validation, then test once)
| Metric | Baseline | Our model |
|---|---|---|
| PR-AUC | ____ | ____ |
| Churners in top 100 | ____ | ____ |

## Intended use
____

## Not for
____

## Assumptions
____

## Risks and limitations
____
