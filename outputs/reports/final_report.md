# World Cup Match Outcome Prediction — Final Report

---

## 1. Problem Definition

Football match outcome prediction is a three-class classification task. For any
given international match, we want to predict one of:

| Class | Meaning |
| --- | --- |
| 0 | Home team loses |
| 1 | Draw |
| 2 | Home team wins |

This is inherently difficult. Football is a low-scoring sport with high outcome
variance: a single set piece or red card can flip a result regardless of
underlying team quality. Factors that matter most — injuries, lineup changes,
tactical choices, fatigue, referee decisions, psychological pressure — are
either unavailable in structured data or unquantifiable before kick-off.

**What a good model looks like.** The naive baseline always predicts the most
frequent class (home win, 47.76% of the test set). Any model that merely learns
"home teams usually win" will achieve ~48% accuracy but fail entirely on losses
and draws. We therefore reject accuracy as our primary metric and adopt
**macro F1**, which gives equal weight to all three classes. A model with macro
F1 of 0.22 (the naive baseline) adds no value; one above 0.45 is genuinely
discriminative.

---

## 2. Dataset

### Sources

Two datasets are used:

- **Kaggle international football results** (`results.csv`): 29,819 matches
  from 1990-01-14 to 2024-03-26 covering all major international competitions.
  Each row records date, home team, away team, scores, tournament name, and
  whether the match was played on neutral ground.
- **Kaggle FIFA World Ranking** (`fifa_ranking-2024-06-20.csv`): 67,472 rows
  covering 216 teams from 1992-12-31 to 2024-06-20. Rankings update roughly
  monthly. Columns used: `rank_date`, `country_full`, `rank`, `total_points`.

### Class imbalance

The target is moderately imbalanced:

| Split | Home loss | Draw | Home win |
| --- | ---: | ---: | ---: |
| Train (1990–2021) | 27.80% | 23.61% | 48.59% |
| Test (2022–2024) | 29.50% | 22.74% | 47.76% |

Home wins are roughly twice as frequent as draws. This imbalance is structural
(home advantage is real in football) and consistent between splits, which is
a good sign. However, if not handled explicitly, any model will over-predict
home wins and essentially ignore draws — the class that is both hardest to
predict and most commercially interesting.

### Train / test split

The split is **strictly chronological** at 2022-01-01: training on 1990–2021,
testing on 2022–2024. This prevents any form of future-data leakage and mimics
real deployment conditions. The 2022–2024 test window is intentionally
challenging: it covers the post-COVID international schedule and the unusual
mid-season Qatar World Cup, both of which disrupted normal preparation cycles.

---

## 3. Champion Approach — Logistic Regression

### Why Logistic Regression

The champion model is **Logistic Regression**. The choice is motivated not by
assumption but by the structure of the problem: football outcome probabilities
are log-linearly related to team quality differences. When two teams are
evenly matched in Elo rating, the log-odds of a home win are near zero; as the
gap grows, log-odds shift proportionally. This is precisely the hypothesis
encoded in a logistic model, and it turns out to hold empirically.

The critical design decision was to engineer features that expose this linear
structure — specifically, Elo ratings — so that LR can exploit it directly
without needing non-linear approximations.

### Feature engineering (29 features)

Every feature is computed from information available **before** the match date.

**Elo ratings (2 features) — the key enabler.**
A self-contained Elo system is built from `results.csv`. Every team starts at
1500. After each match, ratings are updated based on the actual result versus
the expected result (computed from the pre-match rating gap). Parameters:
K = 40 for World Cup, 30 for competitive matches, 15 for friendlies; +100 Elo
bonus applied to the home team when computing expected score on non-neutral
ground. The recorded features — `home_elo` and `away_elo` — are the pre-match
ratings, capturing each team's dynamically updated strength across their
entire competitive history.

Elo is the feature that elevates LR. Because Elo difference is an
approximately linear predictor of win probability in log-odds space, logistic
regression can represent the team quality signal with a single pair of
coefficients — something tree-based models must approximate through many splits.

**Rolling form (16 features).**
For each team and each venue type (home / away), we compute a 5-game rolling
window of win rate, draw rate, goals scored, and goals conceded. Home-game and
away-game records are tracked separately, capturing the asymmetry in team
performance by venue. The `shift(1)` before the window ensures the current
match is never included in its own features.

**FIFA rankings (4 features).**
`home_ranking`, `away_ranking`, `home_points`, and `away_points` joined via
an as-of merge: each match receives the team's most recent published ranking
on or before the match date. While Elo captures dynamic form, FIFA rankings
reflect accumulated prestige over a longer horizon and provide complementary
signal. Teams absent from the ranking (~13% of rows) are imputed to the median
rank (102nd). Derived features (`rank_diff`, `points_diff`) are excluded —
they are exact linear combinations of the individual values and add no
information while inflating apparent feature count.

**Head-to-head record (2 features).**
The home team's historical win rate and draw rate against the specific away
team, computed as a cumulative expanding mean with `shift(1)`. Only the home
team's perspective is retained — the away team's win rate is the arithmetic
complement and perfectly collinear.

**Rest and schedule (2 features).**
Days elapsed since each team's most recent previous match, as a proxy for
fatigue and preparation time.

**Match context (3 features).**
Binary flags for neutral venue, FIFA World Cup, and friendly.

### Handling class imbalance

`class_weight="balanced"` re-weights the logistic loss function so minority
classes count proportionally more. This is what allows the champion to
maintain meaningful recall on draws and home losses without sacrificing overall
macro F1.

### Tuning and validation protocol

Hyperparameter selection (regularisation strength C ∈ {0.1, 0.3, 1.0, 3.0})
uses a chronological 80/20 inner split of the training set, scored by macro
F1. Model stability is assessed with walk-forward cross-validation (test years
2019, 2020, 2021), refitting the scaler inside each fold to prevent leakage.
The 2022+ window is the final held-out test set, never touched during training
or CV.

---

## 4. Challengers

Three challenger models were trained on the same 29 features and evaluated
against the champion. Their role is to verify that Logistic Regression is not
leaving performance on the table.

### Random Forest

Ensemble of 250–400 trees with limited depth (8–12 levels) and minimum leaf
sizes of 1–3. Ensemble averaging reduces variance and naturally handles
non-linear interactions between features — something LR cannot do by
construction.

**Finding:** RF achieves the highest raw accuracy (54.26%) and a test macro F1
of 0.4962, marginally below LR (0.4983). McNemar's test confirms they are
**statistically indistinguishable** (p = 0.13). RF is more computationally
expensive, harder to interpret, and requires calibration for reliable
probability estimates — yet it offers no measurable accuracy advantage over the
champion. Its parity with LR is the strongest single argument for the champion
selection.

### Gradient Boosting

Sequential boosting with 120–220 shallow trees (depth 2–3) and a low learning
rate (0.03–0.10). Each tree corrects the residuals of the ensemble so far.
Boosting is generally the strongest single-model approach for structured
tabular data.

**Finding:** GB leads in walk-forward CV (0.533 ± 0.045) but achieves a test
macro F1 of 0.4955, indistinguishable from LR (p = 0.86). Its marginal CV
advantage disappears within standard deviation overlap. GB is the most opaque
model tested: feature importance scores describe average usage across trees but
cannot explain a specific prediction. Once again, no statistically significant
gain over LR justifies the added complexity.

### Decision Tree

A single decision tree with tuned depth (4–10 levels) and minimum leaf sizes
(10–50 samples). The most interpretable challenger: every prediction can be
traced to a specific decision path, making it fully auditable.

**Finding:** DT is the only model significantly weaker than the ensembles
(p = 0.001 vs RF, p = 0.048 vs GB). Its test macro F1 of 0.4774 and CV score
of 0.501 ± 0.055 confirm that single-tree structure lacks the capacity to
approximate the joint interaction of Elo, form, and rankings. It provides
interpretability at a meaningful performance cost — which is exactly the
tradeoff LR avoids. LR matches the ensembles on performance while equalling
DT on interpretability.

### Summary

| Model | Test Macro F1 | CV Macro F1 | vs LR (McNemar p) |
| --- | ---: | ---: | ---: |
| **Logistic Regression** | **0.4983** | 0.5302 ± 0.0496 | — |
| Random Forest | 0.4962 | 0.5167 ± 0.0563 | 0.13 (n.s.) |
| Gradient Boosting | 0.4955 | 0.5327 ± 0.0454 | 0.86 (n.s.) |
| Decision Tree | 0.4774 | 0.5014 ± 0.0554 | 0.09 (n.s.) |
| Naive baseline | 0.2155 | — | — |

LR, RF, and GB are statistically equivalent. Decision Tree is significantly
weaker than the ensembles. No challenger outperforms the champion. Logistic
Regression is therefore the correct choice: maximum performance with maximum
interpretability.

---

## 5. Results — Data Science Perspective

### Accuracy is no longer the right lens

All four models now exceed the naive accuracy baseline (47.76%), which was not
the case before data enrichment. More importantly, macro F1 exceeds 0.47 for
all models — more than double the naive baseline's 0.22. This jump came from
three compounding decisions: macro-F1 tuning (not accuracy), class balancing,
and progressive feature enrichment.

### Feature contribution by enrichment step

| Feature set | Best test macro F1 | Gain |
| --- | ---: | ---: |
| Form + context (19 features) | 0.4300 | — |
| + FIFA rankings, H2H, rest (27 features) | 0.4855 | +0.0555 |
| + Elo ratings (29 features) | 0.4983 | +0.0128 |

FIFA rankings and Elo are the dominant signal sources. Form features capture
recent-cycle variation that quality metrics smooth over. H2H and rest features
add consistent but smaller incremental value.

### Per-class performance (Logistic Regression)

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Home loss (0) | ~0.49 | ~0.57 | ~0.53 |
| Draw (1) | ~0.32 | ~0.34 | ~0.33 |
| Home win (2) | ~0.65 | ~0.55 | ~0.60 |

Draws remain the hardest class. Draw recall of ~34% is a dramatic improvement
over the near-zero recall of accuracy-tuned models, but it still means the
model misses two out of three draws — a structural reflection of football's
inherent randomness at balanced quality levels. Home wins and losses are better
predicted because the Elo gap is larger and the result more determined.

### Walk-forward CV and temporal stability

CV macro F1 (~0.53) consistently exceeds the test-set score (~0.50) for all
models. This is most plausibly explained by the test window (2022–2024) being
an atypically disrupted period — post-COVID schedules and the mid-season World
Cup — rather than overfitting. The CV folds (2019–2021) represent more typical
international football calendars.

---

## 6. Results — Business / Application Perspective

### What macro F1 ≈ 0.50 means in practice

A macro F1 of 0.50 means the model correctly identifies roughly half of all
home losses, draws, and home wins — each independently, not as a blended
average. In a sport where professional tipsters rarely exceed 55% accuracy on
three-outcome markets, a calibrated probabilistic model at this performance
level has genuine commercial utility.

### Deployment use cases

**Tournament simulation.** The model's three-class probabilities can drive a
Monte Carlo simulation of a tournament bracket. Each match is drawn from the
predicted distribution; running thousands of simulations produces realistic
finish-order distributions and upset probabilities. The Platt-scaled RF
probabilities saved in `rf_calibrated_probs.npy` are suitable for this purpose
— raw `predict_proba` outputs from tree models are often poorly calibrated near
the extremes.

**Pre-match odds benchmarking.** Bookmakers publish implied probabilities
derived from their odds. The model's predicted probabilities can be compared
against the market to identify systematic deviations — matches where the model
is confident and the market disagrees. This provides a principled starting
point for quantitative sports analysis, though it requires careful backtesting
and does not guarantee profitability.

**Scouting and preparation support.** National team analysts can use the Elo
ratings and form features to quantify how much home advantage matters for
specific opponents, or to identify teams that are outperforming or
underperforming their underlying quality. A rising Elo with declining FIFA
ranking points, for instance, signals a team in better shape than their
official standing suggests.

### Why interpretability matters

In a business or editorial context, a prediction must be explainable. A
journalist, federation official, or data analyst will ask: "Why does the model
give France a 65% win probability?" Logistic Regression answers this directly:
because France's Elo exceeds their opponent's by X points, their recent
home-game form is strong, and this is a competitive fixture. The coefficient
signs and magnitudes are stable and auditable.

Random Forest and Gradient Boosting cannot provide this per-prediction
narrative. This is why the champion choice is not merely a performance
decision — it is a deployability decision.

### Limitations for live deployment

- **No live data.** The model uses only pre-match features. It knows nothing
  about the starting lineup, injury list, or travel time — information that
  moves betting markets substantially in the hours before kick-off.
- **Elo warm-up.** Ratings initialised at 1500 in 1990 require several years
  to converge. Predictions for rarely-active or newly promoted nations will
  be noisier than for established footballing countries.
- **Non-stationarity.** Football evolves. The model should be retrained
  periodically and monitored for degrading performance as tactical and
  scheduling norms shift.
- **Draws remain uncertain.** A draw recall of ~34% means the model misses
  two out of three draws. Any product built around draw prediction should
  communicate this uncertainty clearly to end users.

---

## 7. Conclusion

This project built a complete, production-ready ML pipeline for international
football outcome prediction. Three findings stand out.

**Feature design determines champion selection.** Adding Elo ratings — a
per-match, self-contained quality signal computed from the same dataset — was
the single decision that enabled Logistic Regression to match ensemble methods.
Without Elo, LR trailed RF and GB. With it, all three models became
statistically indistinguishable, and the interpretability argument for LR
became decisive.

**Metric design determines model behaviour.** Switching from accuracy to macro
F1 as the tuning objective — and adding class balancing — transformed draw
recall from near-zero to ~34%. This is the difference between a model that
pretends draws do not exist and one that can genuinely discriminate between
all three outcomes. The right evaluation metric is not cosmetic; it reshapes
the model's learned behaviour.

**Complexity does not imply performance.** Random Forest and Gradient Boosting
are more powerful model families than Logistic Regression by construction —
they can approximate arbitrary non-linear functions. Yet on this problem, with
the right features, they offer no measurable advantage. The three challengers
collectively confirm that the champion is not a compromise but the correct
choice: equal performance, full interpretability, and lower inference cost.