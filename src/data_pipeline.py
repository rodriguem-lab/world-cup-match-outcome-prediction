import pandas as pd
import numpy as np
import joblib
from collections import defaultdict
from pathlib import Path
from sklearn.preprocessing import StandardScaler

Path("data/processed").mkdir(parents=True, exist_ok=True)

# =========================
# 1. LOAD & CLEAN DATA
# =========================

df = pd.read_csv("data/raw/results.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df[df["date"] >= "1990-01-01"].copy()
df = df[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]].copy()
df = df.dropna(subset=["home_score", "away_score"])

# =========================
# 2. TARGET VARIABLE
# =========================
# 0 = home loss, 1 = draw, 2 = home win

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 2
    elif row["home_score"] == row["away_score"]:
        return 1
    else:
        return 0

df["target"] = df.apply(get_result, axis=1)

# =========================
# 3. ELO RATINGS
# =========================
# Elo is computed sequentially: pre-match ratings are recorded as features,
# then ratings are updated from the actual result.
# Home advantage: +100 Elo points applied when computing expected score on
# non-neutral ground (neutral is already a separate feature).
# K-factor: 40 for World Cup, 15 for friendlies, 30 for all other matches.

INITIAL_ELO = 1500.0
HOME_ADV     = 100.0

home_teams  = df["home_team"].values
away_teams  = df["away_team"].values
neutrals    = df["neutral"].values
targets     = df["target"].values
tournaments = df["tournament"].values

elo = defaultdict(lambda: INITIAL_ELO)
home_elo_arr = np.empty(len(df))
away_elo_arr = np.empty(len(df))

for i in range(len(df)):
    h, a = home_teams[i], away_teams[i]
    h_elo, a_elo = elo[h], elo[a]
    home_elo_arr[i] = h_elo
    away_elo_arr[i] = a_elo

    h_elo_adj = h_elo + (0.0 if neutrals[i] else HOME_ADV)
    E_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo_adj) / 400.0))

    t = targets[i]
    S_h = 1.0 if t == 2 else (0.5 if t == 1 else 0.0)

    trn = tournaments[i]
    K = 40.0 if trn == "FIFA World Cup" else (15.0 if trn == "Friendly" else 30.0)

    delta = K * (S_h - E_h)
    elo[h] += delta
    elo[a] -= delta

df["home_elo"] = home_elo_arr
df["away_elo"] = away_elo_arr

# =========================
# 4. VECTORIZED ROLLING FORM FEATURES  (renumbered; was §3)
# =========================
# Each team's form is tracked separately for home games and away games.
# Features named: <role>_team_<venue>_<stat>

home_apps = df[["date", "home_team", "home_score", "away_score"]].copy()
home_apps.columns = ["date", "team", "scored", "conceded"]
home_apps["venue"] = "home"

away_apps = df[["date", "away_team", "away_score", "home_score"]].copy()
away_apps.columns = ["date", "team", "scored", "conceded"]
away_apps["venue"] = "away"

apps = pd.concat([home_apps, away_apps], ignore_index=True)
apps = apps.sort_values(["team", "venue", "date"]).reset_index(drop=True)

apps["won"]   = (apps["scored"] > apps["conceded"]).astype(float)
apps["drawn"] = (apps["scored"] == apps["conceded"]).astype(float)

for col in ["won", "drawn", "scored", "conceded"]:
    apps[f"{col}_roll5"] = (
        apps.groupby(["team", "venue"])[col]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

roll_cols = ["won_roll5", "drawn_roll5", "scored_roll5", "conceded_roll5"]
apps[roll_cols] = apps[roll_cols].fillna(0.0)

home_venue = (
    apps[apps["venue"] == "home"][["date", "team"] + roll_cols]
    .drop_duplicates(subset=["date", "team"], keep="first")
)
away_venue = (
    apps[apps["venue"] == "away"][["date", "team"] + roll_cols]
    .drop_duplicates(subset=["date", "team"], keep="first")
)

df = df.merge(
    home_venue.rename(columns={
        "team": "home_team",
        "won_roll5":      "home_team_home_win_rate",
        "drawn_roll5":    "home_team_home_draw_rate",
        "scored_roll5":   "home_team_home_scored_avg",
        "conceded_roll5": "home_team_home_conceded_avg",
    }),
    on=["date", "home_team"], how="left",
)

df = df.merge(
    away_venue.rename(columns={
        "team": "home_team",
        "won_roll5":      "home_team_away_win_rate",
        "drawn_roll5":    "home_team_away_draw_rate",
        "scored_roll5":   "home_team_away_scored_avg",
        "conceded_roll5": "home_team_away_conceded_avg",
    }),
    on=["date", "home_team"], how="left",
)

df = df.merge(
    home_venue.rename(columns={
        "team": "away_team",
        "won_roll5":      "away_team_home_win_rate",
        "drawn_roll5":    "away_team_home_draw_rate",
        "scored_roll5":   "away_team_home_scored_avg",
        "conceded_roll5": "away_team_home_conceded_avg",
    }),
    on=["date", "away_team"], how="left",
)

df = df.merge(
    away_venue.rename(columns={
        "team": "away_team",
        "won_roll5":      "away_team_away_win_rate",
        "drawn_roll5":    "away_team_away_draw_rate",
        "scored_roll5":   "away_team_away_scored_avg",
        "conceded_roll5": "away_team_away_conceded_avg",
    }),
    on=["date", "away_team"], how="left",
)

df["neutral"]      = df["neutral"].astype(int)
df["is_world_cup"] = (df["tournament"] == "FIFA World Cup").astype(int)
df["is_friendly"]  = (df["tournament"] == "Friendly").astype(int)

form_cols = [c for c in df.columns if any(s in c for s in ["_win_rate", "_draw_rate", "_scored_avg", "_conceded_avg"])]
df[form_cols] = df[form_cols].fillna(0.0)

# =========================
# 4. FIFA RANKING FEATURES
# =========================
# As-of join: match each team to their most recent ranking on or before the match date.
# Teams with no FIFA ranking entry are filled with the median rank (neutral assumption).

NAME_MAP = {
    "Brunei":                           "Brunei Darussalam",
    "Burma":                            "Myanmar",
    "Cape Verde":                       "Cabo Verde",
    "South Korea":                      "Korea Republic",
    "North Korea":                      "Korea DPR",
    "USA":                              "United States",
    "Czech Republic":                   "Czechia",
    "DR Congo":                         "Congo DR",
    "Ivory Coast":                      "Côte d'Ivoire",
    "Iran":                             "IR Iran",
    "Kyrgyzstan":                       "Kyrgyz Republic",
    "Laos":                             "Lao PDR",
    "Swaziland":                        "Eswatini",
    "Trinidad & Tobago":                "Trinidad and Tobago",
    "St. Kitts and Nevis":              "Saint Kitts and Nevis",
    "St. Lucia":                        "Saint Lucia",
    "St. Vincent and the Grenadines":   "Saint Vincent and the Grenadines",
}

rankings = pd.read_csv("data/raw/fifa_ranking-2024-06-20.csv")
rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
rankings = (
    rankings[["rank_date", "country_full", "rank", "total_points"]]
    .sort_values("rank_date")
    .reset_index(drop=True)
)

median_rank = float(rankings["rank"].median())

df["home_team_rnk"] = df["home_team"].map(lambda x: NAME_MAP.get(x, x))
df["away_team_rnk"] = df["away_team"].map(lambda x: NAME_MAP.get(x, x))

# df is sorted by date (merge_asof requires left to be sorted on the join key)
home_rnk = rankings.rename(columns={
    "country_full": "home_team_rnk",
    "rank":         "home_ranking",
    "total_points": "home_points",
})[["rank_date", "home_team_rnk", "home_ranking", "home_points"]]

df = pd.merge_asof(
    df, home_rnk,
    left_on="date", right_on="rank_date",
    by="home_team_rnk", direction="backward",
)
df = df.drop(columns=["rank_date"], errors="ignore")

away_rnk = rankings.rename(columns={
    "country_full": "away_team_rnk",
    "rank":         "away_ranking",
    "total_points": "away_points",
})[["rank_date", "away_team_rnk", "away_ranking", "away_points"]]

df = pd.merge_asof(
    df, away_rnk,
    left_on="date", right_on="rank_date",
    by="away_team_rnk", direction="backward",
)
df = df.drop(columns=["rank_date", "home_team_rnk", "away_team_rnk"], errors="ignore")

df["home_ranking"] = df["home_ranking"].fillna(median_rank)
df["away_ranking"] = df["away_ranking"].fillna(median_rank)
df["home_points"]  = df["home_points"].fillna(0.0)
df["away_points"]  = df["away_points"].fillna(0.0)

# =========================
# 5. HEAD-TO-HEAD FEATURES
# =========================
# For each match, compute the home team's historical win/draw rate against the
# specific away team across ALL past meetings (any venue). Only the home team's
# perspective is included; the away team's rates are collinear (win + draw + loss = 1).

h2h_home = df[["date", "home_team", "away_team", "target"]].copy()
h2h_home.columns = ["date", "team", "opponent", "target"]
h2h_home["h2h_won"]   = (h2h_home["target"] == 2).astype(float)
h2h_home["h2h_drawn"] = (h2h_home["target"] == 1).astype(float)

h2h_away = df[["date", "away_team", "home_team", "target"]].copy()
h2h_away.columns = ["date", "team", "opponent", "target"]
h2h_away["h2h_won"]   = (h2h_away["target"] == 0).astype(float)
h2h_away["h2h_drawn"] = (h2h_away["target"] == 1).astype(float)

h2h = pd.concat([
    h2h_home[["date", "team", "opponent", "h2h_won", "h2h_drawn"]],
    h2h_away[["date", "team", "opponent", "h2h_won", "h2h_drawn"]],
], ignore_index=True)
h2h = h2h.sort_values(["team", "opponent", "date"]).reset_index(drop=True)

h2h["h2h_win_rate"]  = h2h.groupby(["team", "opponent"])["h2h_won"].transform(
    lambda s: s.shift(1).expanding().mean()
)
h2h["h2h_draw_rate"] = h2h.groupby(["team", "opponent"])["h2h_drawn"].transform(
    lambda s: s.shift(1).expanding().mean()
)
h2h[["h2h_win_rate", "h2h_draw_rate"]] = h2h[["h2h_win_rate", "h2h_draw_rate"]].fillna(0.0)

h2h_for_home = (
    h2h[["date", "team", "opponent", "h2h_win_rate", "h2h_draw_rate"]]
    .rename(columns={
        "team": "home_team", "opponent": "away_team",
        "h2h_win_rate":  "home_h2h_win_rate",
        "h2h_draw_rate": "home_h2h_draw_rate",
    })
    .drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
)
df = df.merge(h2h_for_home, on=["date", "home_team", "away_team"], how="left")
df[["home_h2h_win_rate", "home_h2h_draw_rate"]] = df[
    ["home_h2h_win_rate", "home_h2h_draw_rate"]].fillna(0.0)

# =========================
# 6. DAYS SINCE LAST MATCH
# =========================
# Days elapsed since each team's most recent previous match (any venue).
# Proxy for rest/fatigue and schedule congestion.

all_apps = (
    apps[["team", "date"]]
    .drop_duplicates()
    .sort_values(["team", "date"])
    .reset_index(drop=True)
)
all_apps["last_match_date"] = all_apps.groupby("team")["date"].shift(1)
all_apps["days_since_last"] = (all_apps["date"] - all_apps["last_match_date"]).dt.days
all_apps["days_since_last"] = all_apps["days_since_last"].fillna(0.0)

days_lookup = all_apps[["team", "date", "days_since_last"]].drop_duplicates(subset=["team", "date"], keep="first")

df = df.merge(
    days_lookup.rename(columns={"team": "home_team", "days_since_last": "home_days_since_last"}),
    on=["date", "home_team"], how="left",
)
df = df.merge(
    days_lookup.rename(columns={"team": "away_team", "days_since_last": "away_days_since_last"}),
    on=["date", "away_team"], how="left",
)
df[["home_days_since_last", "away_days_since_last"]] = df[
    ["home_days_since_last", "away_days_since_last"]].fillna(0.0)

clean_df = df.copy()

# =========================
# 7. REMOVE UNINFORMATIVE ROWS
# =========================
# Drop rows where BOTH teams have zero historical data at ALL venues.

both_no_info = (
    (clean_df["home_team_home_win_rate"]   == 0) &
    (clean_df["home_team_home_scored_avg"] == 0) &
    (clean_df["home_team_away_win_rate"]   == 0) &
    (clean_df["home_team_away_scored_avg"] == 0) &
    (clean_df["away_team_home_win_rate"]   == 0) &
    (clean_df["away_team_home_scored_avg"] == 0) &
    (clean_df["away_team_away_win_rate"]   == 0) &
    (clean_df["away_team_away_scored_avg"] == 0)
)
clean_df = clean_df[~both_no_info].reset_index(drop=True)

# =========================
# 8. TRAIN / TEST SPLIT
# =========================

split_date = pd.Timestamp("2022-01-01")
train = clean_df[clean_df["date"] < split_date].copy()
test  = clean_df[clean_df["date"] >= split_date].copy()

feature_cols = [
    # Home team: form at home venue (last <=5 home games before this match)
    "home_team_home_win_rate",
    "home_team_home_draw_rate",
    "home_team_home_scored_avg",
    "home_team_home_conceded_avg",
    # Home team: form at away venue (last <=5 away games before this match)
    "home_team_away_win_rate",
    "home_team_away_draw_rate",
    "home_team_away_scored_avg",
    "home_team_away_conceded_avg",
    # Away team: form at home venue
    "away_team_home_win_rate",
    "away_team_home_draw_rate",
    "away_team_home_scored_avg",
    "away_team_home_conceded_avg",
    # Away team: form at away venue
    "away_team_away_win_rate",
    "away_team_away_draw_rate",
    "away_team_away_scored_avg",
    "away_team_away_conceded_avg",
    # Match context
    "neutral",
    "is_world_cup",
    "is_friendly",
    # Elo ratings (pre-match, updated after every result)
    "home_elo",
    "away_elo",
    # FIFA rankings (as of match date)
    "home_ranking",
    "away_ranking",
    "home_points",
    "away_points",
    # Head-to-head (home team's full historical record vs this opponent)
    "home_h2h_win_rate",
    "home_h2h_draw_rate",
    # Rest / schedule congestion
    "home_days_since_last",
    "away_days_since_last",
]

X_train_raw = train[feature_cols].values
X_test_raw  = test[feature_cols].values
y_train = train["target"].values
y_test  = test["target"].values

# =========================
# 9. STANDARDIZATION
# =========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

joblib.dump(scaler, "data/processed/scaler.joblib")

# =========================
# 10. SAVE OUTPUTS
# =========================

clean_df.to_csv("data/processed/clean_matches.csv", index=False)
np.save("data/processed/X_train.npy", X_train_scaled)
np.save("data/processed/X_test.npy",  X_test_scaled)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy",  y_test)

with open("data/processed/feature_names.txt", "w") as f:
    for col in feature_cols:
        f.write(col + "\n")

rnk_fill_home = (clean_df["home_ranking"] == median_rank).mean()
rnk_fill_away = (clean_df["away_ranking"] == median_rank).mean()

summary = f"""
DATA SUMMARY

Total matches after cleaning: {len(clean_df)}
Date range: {clean_df['date'].min()} to {clean_df['date'].max()}

Train period: {train['date'].min()} to {train['date'].max()}
Train size:   {len(train)}

Test period: {test['date'].min()} to {test['date'].max()}
Test size:   {len(test)}

Target: 0 = home loss | 1 = draw | 2 = home win

Class distribution (train):
{pd.Series(y_train).value_counts(normalize=True).sort_index().to_string()}

Class distribution (test):
{pd.Series(y_test).value_counts(normalize=True).sort_index().to_string()}

Features ({len(feature_cols)} total):
{chr(10).join(f'  {c}' for c in feature_cols)}

FIFA ranking imputed to median ({median_rank:.0f}) — home team: {rnk_fill_home:.1%}
FIFA ranking imputed to median ({median_rank:.0f}) — away team: {rnk_fill_away:.1%}
"""

with open("data/processed/data_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("Pipeline complete.")
