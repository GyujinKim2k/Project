import requests
import csv
import datetime
import time
from tqdm import tqdm

def get_historical_daily(stn_id, date, service_key):
    url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
    params = {
        'ServiceKey': service_key,
        'pageNo': '1',
        'numOfRows': '10',
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'DAY',
        'startDt': date,
        'endDt': date,
        'stnIds': str(stn_id)
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    items = resp.json()['response']['body']['items']['item']
    if not items:
        raise ValueError(f"No data for {date}")
    d = items[0]
    p_raw = d.get('sumRn')
    if p_raw == '':
        p_raw = 0
    else:
        p_raw = float(p_raw)
    w_raw = d.get('avgWs')
    if w_raw == '':
        w_raw = 2
    else:
        w_raw = float(w_raw)

    return {
        'date': date,
        'T':   float(d.get('avgTa', 0)),
        'RH':  float(d.get('avgRhm', 0)),
        'P':   float(p_raw),
        'W':   float(w_raw),
        'S':   float(d.get('ssDur', 0)),
    }

def save_weather_csv_with_retry(start_date, end_date, station_id, service_key, filename):
    # 1) build YYYYMMDD list
    sd = datetime.date.fromisoformat(start_date)
    ed = datetime.date.fromisoformat(end_date)
    dates = [(sd + datetime.timedelta(days=i)).strftime('%Y%m%d')
             for i in range((ed - sd).days + 1)]

    # 2) open CSV
    fieldnames = ['date','T','RH','P','W','S']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 3) iterate with tqdm + retry
        for dt in tqdm(dates, desc='Fetching weather'):
            while True:
                try:
                    row = get_historical_daily(station_id, dt, service_key)
                    break
                except Exception as e:
                    print(f"[{dt}] error: {e} — retrying in 5 s")
                    time.sleep(5)

            writer.writerow(row)
            time.sleep(0.2)   # gentle pause between calls

if __name__ == '__main__':
    SERVICE_KEY = 'ovbSd4fqkM8YCBsJZb3IXRVetj2J5nfwIfpFULsUyXn7nirAutYIrpAaLxFIG6bHP3mbSp203cz1k385t8AbYQ=='
    save_weather_csv_with_retry(
        start_date='2023-01-01',
        end_date  ='2023-12-31',
        station_id=108,
        service_key=SERVICE_KEY,
        filename  ='weather_2023.csv'
    )
    print("✅ Done: weather_2023.csv written.")
