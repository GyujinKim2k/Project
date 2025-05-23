import csv
import weatherScore
import importlib


def get_weather_score_from_csv(date, csv_file='data/weather/weather_2023.csv'):
    """
    date: str in 'YYYYMMDD' format
    csv_file: path to a CSV with columns date,T,RH,P,W,S
    returns: float weather score for that date
    """
    importlib.reload(weatherScore)
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['date'] == date:
                T  = float(row['T'])
                RH = float(row['RH'])
                P  = float(row['P'])
                W  = float(row['W'])
                S  = float(row['S'])
                return weatherScore.calculate_weather_score(T, RH, P, W, S)
    raise KeyError(f"No data for date {date} in {csv_file}")

# 예시 사용
if __name__ == '__main__':
    print(get_weather_score_from_csv('20230512'))  # e.g. 78.3
    print(get_weather_score_from_csv('20231231'))
