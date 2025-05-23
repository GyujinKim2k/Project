# day_factor_with_annual.py

import numpy as np
import pandas as pd
from prophet.serialize import model_from_json

# 1) Load your saved Prophet model (exported via m.model_to_json())
with open('data/prophet_model.json', 'r') as fin:
    m = model_from_json(fin.read())

# 2) Regression parameters (from your log-log OLS)
b0   = 10.788757745111797
bW   = 1.0211359015678163
bH   = 0.8709493703867706
bWH  = -2.5627887775938483


# 3) Prophet weekday multipliers
weekly_multiplier = {
    0: 0.930774, 1: 0.879771, 2: 0.868360,
    3: 0.876122, 4: 0.961106, 5: 1.250882,
    6: 1.232985
}

def holiday_mult(run_length):
    if run_length >= 3:
        return 1.50
    elif run_length >= 1:
        return 1.30
    else:
        return 1.00

def get_annual_multiplier(date):
    """
    Returns the Prophet multiplicative 'yearly' effect for a single date.
    """
    df = pd.DataFrame({'ds': [pd.to_datetime(date)]})
    seas = m.predict_seasonal_components(df)
    # multiplicative mode: seas['yearly'] is (factor − 1)
    return float(seas['yearly'].iloc[0] + 1)

def get_day_factor(input_date):
    """
    Args:
      input_date: str or pd.Timestamp
    Returns:
      predicted detrended_rev_per_vehicle including 
      weekday, holiday-length and annual-season effects
    """
    date = pd.to_datetime(input_date).normalize()

    # --- build offday_run exactly as you had before ---
    df_hol = pd.read_feather('data/2023~2025년_휴일_데이터_38rows.feather')
    df_hol = df_hol.rename(columns={'일자':'date','휴일명':'holiday_name'})
    df_hol['date'] = pd.to_datetime(df_hol['date']).dt.normalize()

    cal = pd.DataFrame({'date': pd.date_range(df_hol['date'].min(),
                                              df_hol['date'].max())})
    cal['is_weekend'] = cal['date'].dt.weekday >= 5
    cal['is_holiday'] = cal['date'].isin(df_hol['date'])
    cal['is_offday']  = cal['is_holiday']
    cal['block_id']   = (cal['is_offday'] != cal['is_offday'].shift(1)).cumsum()
    cal['offday_run'] = cal.groupby('block_id')['is_offday'].transform('sum')
    cal['offday_run'] = cal['offday_run'].where(cal['is_offday'], 0)

    run = int(cal.loc[cal['date'] == date, 'offday_run'])
    #print(f"offday_run: {run}")
    # --- weekday & holiday factors ---
    wf = weekly_multiplier[date.weekday()]
    hf = holiday_mult(run)

    # --- base log-log prediction ---
    lnW = np.log(wf)
    lnH = np.log(hf)
    lnY = b0 + bW*lnW + bH*lnH + bWH*lnW*lnH
    base = float(np.exp(lnY))

    # --- annual multiplier ---
    annual = get_annual_multiplier(date)

    return base * annual

if __name__ == "__main__":
    # simple smoke test
    for d in ['2023-01-06','2023-05-05','2023-12-25']:
        print(d, get_day_factor(d))
