import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("results.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Keep modern football only
df = df[df["date"] >= "1990-01-01"].copy()

# Keep only useful columns
df = df[
    [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    ]
].copy()

# Remove missing scores
df = df.dropna(subset=["home_score", "away_score"])

# =========================
# 2. CREATE TARGET
# =========================
# 0 = home loss
# 1 = draw
# 2 = home win

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] == row["away_score"]:
        return 1
    else:
        return 0


df["target"] = df.apply(get_result, axis=1)

# =========================
# 3. FEATURE ENGINEERING
# =========================

def get_team_past_matches(data, team, current_date):
    past_matches = data[
        (
            ((data["home_team"] == team) | (data["away_team"] == team))
            & (data["date"] < current_date)
        )
    ]
    return past_matches.tail(5)


def compute_team_features(data, team, current_date):
    past = get_team_past_matches(data, team, current_date)

    if len(past) == 0:
        return {
            "win_rate_5": 0.0,
            "draw_rate_5": 0.0,
            "loss_rate_5": 0.0,
            "goals_scored_avg_5": 0.0,
            "goals_conceded_avg_5": 0.0,
            "goal_diff_avg_5": 0.0,
        }

    wins = 0
    draws = 0
    losses = 0
    goals_scored = []
    goals_conceded = []

    for _, match in past.iterrows():
        if match["home_team"] == team:
            scored = match["home_score"]
            conceded = match["away_score"]
        else:
            scored = match["away_score"]
            conceded = match["home_score"]

        goals_scored.append(scored)
        goals_conceded.append(conceded)

        if scored > conceded:
            wins += 1
        elif scored == conceded:
            draws += 1
        else:
            losses += 1

    n = len(past)

    return {
        "win_rate_5": wins / n,
        "draw_rate_5": draws / n,
        "loss_rate_5": losses / n,
        "goals_scored_avg_5": np.mean(goals_scored),
        "goals_conceded_avg_5": np.mean(goals_conceded),
        "goal_diff_avg_5": np.mean(np.array(goals_scored) - np.array(goals_conceded)),
    }


features = []

for idx, row in df.iterrows():
    home_team = row["home_team"]
    away_team = row["away_team"]
    current_date = row["date"]

    home_features = compute_team_features(df, home_team, current_date)
    away_features = compute_team_features(df, away_team, current_date)

    feature_row = {
        "date": current_date,
        "home_team": home_team,
        "away_team": away_team,

        "home_win_rate_5": home_features["win_rate_5"],
        "home_draw_rate_5": home_features["draw_rate_5"],
        "home_loss_rate_5": home_features["loss_rate_5"],
        "home_goals_scored_avg_5": home_features["goals_scored_avg_5"],
        "home_goals_conceded_avg_5": home_features["goals_conceded_avg_5"],
        "home_goal_diff_avg_5": home_features["goal_diff_avg_5"],

        "away_win_rate_5": away_features["win_rate_5"],
        "away_draw_rate_5": away_features["draw_rate_5"],
        "away_loss_rate_5": away_features["loss_rate_5"],
        "away_goals_scored_avg_5": away_features["goals_scored_avg_5"],
        "away_goals_conceded_avg_5": away_features["goals_conceded_avg_5"],
        "away_goal_diff_avg_5": away_features["goal_diff_avg_5"],

        "win_rate_difference": home_features["win_rate_5"] - away_features["win_rate_5"],
        "goal_diff_difference": home_features["goal_diff_avg_5"] - away_features["goal_diff_avg_5"],

        "neutral": int(row["neutral"]),
        "is_world_cup": int(row["tournament"] == "FIFA World Cup"),
        "is_friendly": int(row["tournament"] == "Friendly"),

        "target": row["target"],
    }

    features.append(feature_row)

clean_df = pd.DataFrame(features)

# =========================
# 4. REMOVE EARLY UNINFORMATIVE ROWS
# =========================

# Remove rows where both teams have zero historical information
clean_df = clean_df[
    ~(
        (clean_df["home_win_rate_5"] == 0)
        & (clean_df["away_win_rate_5"] == 0)
        & (clean_df["home_goals_scored_avg_5"] == 0)
        & (clean_df["away_goals_scored_avg_5"] == 0)
    )
].reset_index(drop=True)

# =========================
# 5. TRAIN / TEST SPLIT
# =========================

split_date = pd.Timestamp("2022-01-01")

train = clean_df[clean_df["date"] < split_date].copy()
test = clean_df[clean_df["date"] >= split_date].copy()

feature_cols = [
    "home_win_rate_5",
    "home_draw_rate_5",
    "home_loss_rate_5",
    "home_goals_scored_avg_5",
    "home_goals_conceded_avg_5",
    "home_goal_diff_avg_5",
    "away_win_rate_5",
    "away_draw_rate_5",
    "away_loss_rate_5",
    "away_goals_scored_avg_5",
    "away_goals_conceded_avg_5",
    "away_goal_diff_avg_5",
    "win_rate_difference",
    "goal_diff_difference",
    "neutral",
    "is_world_cup",
    "is_friendly",
]

X_train = train[feature_cols].values
X_test = test[feature_cols].values
y_train = train["target"].values
y_test = test["target"].values

# =========================
# 6. STANDARDIZATION
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 7. SAVE FILES
# =========================

clean_df.to_csv("clean_matches.csv", index=False)

np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy", X_test_scaled)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

with open("feature_names.txt", "w") as f:
    for col in feature_cols:
        f.write(col + "\n")

summary = f"""
DATA SUMMARY

Total matches after cleaning: {len(clean_df)}

Date range:
{clean_df['date'].min()} to {clean_df['date'].max()}

Train period:
{train['date'].min()} to {train['date'].max()}
Train size: {len(train)}

Test period:
{test['date'].min()} to {test['date'].max()}
Test size: {len(test)}

Target meaning:
0 = home team loses
1 = draw
2 = home team wins

Class distribution train:
{pd.Series(y_train).value_counts(normalize=True).sort_index()}

Class distribution test:
{pd.Series(y_test).value_counts(normalize=True).sort_index()}

Features used:
{feature_cols}
"""

with open("data_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("Files successfully created.")