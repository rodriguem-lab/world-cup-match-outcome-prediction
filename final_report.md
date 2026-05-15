# World Cup Match Outcome Prediction Project

## 1. Problem Definition

Football match prediction is a difficult machine learning problem because match
outcomes are noisy, competitive, and affected by factors that are not always
available in structured datasets, such as injuries, tactical choices, referee
decisions, fatigue, and psychological pressure. The objective of this project is
to predict the outcome of international football matches as a three-class
classification task:

| Class | Meaning |
| --- | --- |
| 0 | Home team loses |
| 1 | Draw |
| 2 | Home team wins |

The main research question is:

> Can machine learning predict international football match outcomes using
> historical match data, and which model performs best?

This problem is relevant for sports analytics, betting market research, team
strategy, fan engagement, and tournament forecasting.

To make the evaluation more meaningful, the project also includes a naive benchmark. This benchmark always predicts the most frequent outcome observed in the training set. It gives us a simple reference point to check whether the machine learning models actually add predictive value.

## 2. Dataset Description and Preprocessing

The project uses the Kaggle international football results dataset. The raw data
contains historical international matches with match dates, home and away teams,
scores, tournament type, and neutral venue information. The final pipeline keeps
modern football only, starting from 1990, to reduce the impact of very old match
patterns that are less representative of current football.

The preprocessing steps were:

1. Load `results.csv`.
2. Convert `date` to datetime.
3. Sort matches chronologically.
4. Keep useful columns only.
5. Remove matches with missing scores.
6. Create the target variable:
   - 0 = home loss
   - 1 = draw
   - 2 = home win
7. Compute all rolling features using only matches played before the prediction
   date.
8. Split the data chronologically:
   - Train = matches before 2022
   - Test = matches from 2022 onward
9. Standardize numerical features with `StandardScaler`.

Final dataset summary:

| Item | Value |
| --- | ---: |
| Total matches after cleaning | 29,910 |
| Date range | 1990-01-14 to 2024-03-26 |
| Train size | 27,588 |
| Test size | 2,322 |
| Number of features | 17 |

Class distribution:

| Split | Home loss | Draw | Home win |
| --- | ---: | ---: | ---: |
| Train | 27.80% | 23.62% | 48.58% |
| Test | 29.50% | 22.74% | 47.76% |

The engineered features include recent win, draw, and loss rates over the last
five matches, average goals scored, average goals conceded, average goal
difference, comparative form differences, neutral venue, World Cup indicator,
and friendly match indicator. Because every feature is computed from past
matches only, the pipeline avoids data leakage.

## 3. Benchmark and Champion Approach
Before selecting the champion model, we evaluated a naive benchmark based on the most frequent class strategy. This model always predicts the outcome that appears most often in the training set. Since home wins are the most frequent class, the benchmark mainly shows how far a very simple rule can go without learning from the engineered features.

Logistic Regression was selected as the champion machine learning model because it is simple, fast, interpretable, and academically easy to explain. It learns a linear relationship between the engineered features and the probability of each match outcome.

Strengths:

- Easy to interpret and reproduce.
- Fast to train.
- Useful baseline for comparing more complex models.

Weaknesses:

- Limited ability to capture non-linear interactions.
- Football outcomes are noisy and may not follow linear decision boundaries.
- Draw prediction is especially difficult.

The best Logistic Regression parameter selected by chronological validation was:

```text
C = 1.0
```

## 4. Challenger Models

### Decision Tree

Decision Tree was used as the first non-linear challenger. It learns simple
if-then rules from the engineered features, making it easy to interpret. This
model tests whether rule-based splits on recent form can outperform the linear
champion model.

Best selected parameters:

```text
max_depth = 4
min_samples_leaf = 10
```

### Random Forest

Random Forest was used as a non-linear ensemble challenger. It combines many
decision trees and can capture interactions between team form, goal difference,
tournament context, and neutral venue information.

Best selected parameters:

```text
n_estimators = 250
max_depth = 8
min_samples_leaf = 1
```

### Gradient Boosting Classifier

The original plan recommended XGBoost. Since `xgboost` was not available in the
local Python environment, the project used scikit-learn's
`GradientBoostingClassifier` as the planned fallback boosting model. This still
represents the same challenger family: sequential boosted trees that try to
correct previous errors.

Best selected parameters:

```text
n_estimators = 120
learning_rate = 0.05
max_depth = 2
```

## 5. Data Science Results

The models were evaluated on the chronological test set from 2022 onward.

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Baseline | 47.76% | 0.2155 | 0.3088 |
| Logistic Regression | 53.10% | 0.3739 | 0.4517 |
| Decision Tree | 52.89% | 0.3569 | 0.4371 |
| Random Forest | 52.93% | 0.3641 | 0.4437 |
| Gradient Boosting | 52.93% | 0.3697 | 0.4481 |

Confusion matrices were saved in:

```text
confusion_matrices.png
```

Accuracy comparison chart:

```text
accuracy_comparison.png
```

Feature importance plot:

```text
feature_importance.png
```

The naive benchmark achieved 47.76% accuracy and a macro F1 score of 0.2155. Logistic Regression improved accuracy to 53.10% and macro F1 to 0.3739. This means that the model adds predictive value beyond simply predicting the most frequent outcome.

Among the machine learning models, Logistic Regression achieved the highest accuracy. Decision Tree, Random Forest, and Gradient Boosting were very close, yet none of them delivered a clear improvement over the champion model. For this reason, Logistic Regression remains the preferred model: it is slightly more accurate, simpler to explain, and more transparent.

The confusion matrices show that all models predict home wins more easily than draws. This is an important limitation. Overall accuracy gives a useful first evaluation, but it can hide poor performance on the minority draw class.
Top boosting feature importances:

| Feature | Importance |
| --- | ---: |
| goal_diff_difference | 0.7013 |
| away_goal_diff_avg_5 | 0.0549 |
| home_goal_diff_avg_5 | 0.0524 |
| is_friendly | 0.0452 |
| neutral | 0.0436 |
| away_goals_conceded_avg_5 | 0.0269 |
| away_goals_scored_avg_5 | 0.0227 |
| home_goals_conceded_avg_5 | 0.0227 |

The most important variable was the recent goal difference gap between the home
and away team, which is intuitive because recent scoring strength and defensive
performance are direct indicators of current form.

## 6. Business and Application Results

The results show that machine learning can capture part of the signal in international football outcomes, but the predictive ceiling is limited when using only historical match results and simple contextual indicators.

The champion model improves over the naive benchmark, moving from 47.76% to 53.10% accuracy. This is a real gain, but the model is still not strong enough to be used alone for high risk betting decisions.

Potential applications:

- Sports analytics teams can use the model as a baseline forecasting tool.
- Media platforms can generate pre-match probabilities and fan engagement
  content.
- Tournament simulations can use model probabilities as inputs.
- Betting analysts can combine this type of model with odds, injuries, squads,
  and market information.

Limitations:

- No player-level information.
- No injury, lineup, or fatigue data.
- No FIFA rankings or Elo ratings.
- No betting odds.
- Draws are intrinsically difficult to predict.
- Football contains high randomness due to low scoring.

Future improvements could include Elo ratings, FIFA rankings, squad strength,
player availability, rest days, travel distance, and betting market odds.

## 7. Conclusion

This project built a complete chronological machine learning pipeline for
international football outcome prediction. The final dataset contains 29,910
matches and 17 engineered features based only on past information.
The naive benchmark achieved 47.76% accuracy. Logistic Regression improved this result to 53.10%, with a macro F1 score of 0.3739. The tree based challengers were close, but they did not provide enough improvement to justify replacing the simpler champion model.

The strongest predictive signal was the recent goal difference comparison between the two teams.

The main conclusion is that football outcomes are partially predictable from
historical form, but the sport remains highly uncertain. Better real-world
performance would require richer contextual data beyond match results alone.
