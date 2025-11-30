# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI()
art = joblib.load("./model/dish_top_model.joblib")
clf = art["model"]
le_weekday = art["le_weekday"]
le_season = art["le_season"]
le_tempbucket = art["le_tempbucket"]
le_top_dish = art["le_top_dish"]
date_groups = art["date_groups"]  # historical top per date for averaging


class PredictRequest(BaseModel):
    date: str  # 'YYYY-MM-DD'
    outside_temperature: float = None
    is_raining: int = None  # 0 or 1
    special_event: int = 0
    is_holiday: int = 0


@app.post("/predict")
def predict(req: PredictRequest):
    # parse date
    dt = datetime.strptime(req.date, "%Y-%m-%d")
    weekday = dt.strftime("%A")  # e.g., 'Monday'
    # choose season mapping function (you must adapt mapping if your dataset uses different season labels)
    month = dt.month
    if month in [12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    else:
        season = "autumn"

    # temp fallback: if not provided, use historical average for that day-of-year
    if req.outside_temperature is None:
        # approximate using date_groups median for same weekday
        fallback = date_groups[date_groups["weekday"] == weekday][
            "outside_temperature"
        ].median()
        temp = float(fallback) if not np.isnan(fallback) else 20.0
    else:
        temp = req.outside_temperature

    # bucket temp to same bins used in training
    import pandas as pd

    temp_bucket = pd.cut(
        [temp], bins=[-50, 5, 15, 25, 50], labels=["very_cold", "cold", "mild", "hot"]
    )[0]

    # encode features
    X = pd.DataFrame(
        {
            "weekday": [
                (
                    le_weekday.transform([weekday])[0]
                    if weekday in le_weekday.classes_
                    else 0
                )
            ],
            "season": [
                le_season.transform([season])[0] if season in le_season.classes_ else 0
            ],
            "is_holiday": [int(req.is_holiday)],
            "special_event": [int(req.special_event)],
            "temp_bucket": [
                (
                    le_tempbucket.transform([str(temp_bucket)])[0]
                    if str(temp_bucket) in le_tempbucket.classes_
                    else 0
                )
            ],
            "is_raining": [int(req.is_raining) if req.is_raining is not None else 0],
        }
    )

    pred_idx = clf.predict(X)[0]
    proba = clf.predict_proba(X).max()
    pred_dish = le_top_dish.inverse_transform([pred_idx])[0]

    # expected qty: historical average sold_quantity for this dish on matching weekday/season/is_holiday
    hist = date_groups[date_groups["top_dish"] == pred_dish]
    match = hist[
        (hist["weekday"] == weekday)
        & (hist["season"] == season)
        & (hist["is_holiday"] == req.is_holiday)
    ]
    if len(match) == 0:
        expected_qty = float(hist["top_qty"].mean())
    else:
        expected_qty = float(match["top_qty"].mean())

    return {
        "predicted_dish": pred_dish,
        "confidence": float(proba),
        "expected_sold_quantity": expected_qty,
    }
