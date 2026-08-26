# Machine Learning MLB Salary Prediction

CS 436/536 — Intro to Machine Learning · Final Project

## Problem

Predict a Major League Baseball player's salary for a given season from that season's batting and pitching statistics. Supervised regression, target is `salary` in nominal USD.

## Dataset

[Baseball Databank / Lahman Baseball Database](https://www.kaggle.com/datasets/open-source-sports/baseball-databank)

## Notebooks

- `ML_Final_Project_EDA.ipynb` — EDA and hypothesis generation
- `ML_Final_Project_Full.ipynb` — full report: data prep, models, results, and conclusions

## Approach
Linear regression as a baseline plus Random Forest and HistGradientBoosting, with a small grid search on the training split. Trees get raw data since they handle `NaN` natively; linear regression gets median imputation plus missingness indicator flags. One 80/20 split stratified by player type, reused across every experiment. Metrics are RMSE, MAE, R², and median absolute percentage error.

## Hypotheses and results

**H1 — Separate pitcher/non-pitcher models beat a pooled model.** ❌ Not supported. Splitting by role gained under 1% RMSE for every model (0.2% linear, 0.3% RF, 0.7% boosted). The pipelines already handled the structural missingness, so the split solved a problem that was already solved.

**H2 — Log-transforming salary improves performance.** ✅ Supported, with a tradeoff. RMSE got worse ($2.84M → $3.05M) but MAE improved ($1.79M → $1.52M) and MedAPE dropped from 137% to 73%. The log model prices the typical player better and superstar contracts worse.

**H3 — Year plus playing time matches the full feature set.** ❌ Not supported as stated. The full set beat the two-feature baseline by ~15% RMSE ($2.68M vs $3.15M), so the performance stats do carry real signal. The redundancy half held up though: nine collinear volume columns could be dropped for under 2% cost, and no rate statistic ranked near the top of the permutation importances.

**H4 — Home runs are a weak predictor when roles are pooled, and strengthen when separated.** ✅ Supported. Correlation with salary rose from 0.289 (pooled) to 0.398 (non-pitchers only), with standardized coefficients and Random Forest importance moving the same direction.

## Main finding

Most of the predictable signal here isn't about player quality. `yearID` is worth about $763,000 of RMSE on its own — over 3× any performance column — and the next-most-important features are all playing-time counts. A model on this data earns most of its score by learning "recent season, plays a lot."

Errors also stayed large in absolute terms: the best model reached ~$2.68M RMSE against a median salary near $600K. Part of that is unrecoverable, since low-production seasons belong to rookies, injured veterans, and call-ups whose pay was set by contract status, which this dataset doesn't record.

## Limitations

- Salaries are nominal, not inflation-adjusted, so every importance ranking is partly a statement about the calendar.
- Every comparison used a single 80/20 split, so sub-1% differences could be split noise.
- No service time, free agency status, or contract year — likely a major source of unexplained variance.
