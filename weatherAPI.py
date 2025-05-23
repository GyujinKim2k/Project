
import requests
import weatherScore

def get_historical_daily(stn_id, date, service_key):
    """
    stn_id: int or str, ASOS 지점 번호 (예: 108)
    date: str, 조회 날짜 'YYYYMMDD' (전일까지만 가능)
    service_key: str, 공공데이터포털에서 발급받은 인증키
    """
    url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
    params = {
        'ServiceKey': service_key,
        'pageNo': '1',
        'numOfRows': '10',
        'dataType': 'JSON',
        'dataCd': 'ASOS',      # ASOS: 종관기상관측
        'dateCd': 'DAY',       # DAY: 일자료
        'startDt': date,
        'endDt': date,
        'stnIds': str(stn_id)
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    items = resp.json()['response']['body']['items']['item']
    if not items:
        raise ValueError(f"{date}에 대한 데이터가 없습니다.")
    d = items[0]
    p_raw = d.get('sumRn')
    if p_raw == '':
        p_raw = 0
    else:
        p_raw = float(p_raw)
    return {
        'T':  d.get('avgTa'),   # 평균 기온(℃)
        'RH': d.get('avgRhm'),  # 평균 상대습도(%)
        'P':  p_raw,   # 일강수량(mm)
        'W':  d.get('avgWs'),   # 평균 풍속(m/s)
        'S':  d.get('ssDur')    # 일조시간(h)
    }

def weather_score(query_date, station_id=108):
    
    # 예시: 2025년 5월 12일, 서울(108) 자료
    SERVICE_KEY = 'ovbSd4fqkM8YCBsJZb3IXRVetj2J5nfwIfpFULsUyXn7nirAutYIrpAaLxFIG6bHP3mbSp203cz1k385t8AbYQ=='

    
    #query_date = '20100101'    # YYYYMMDD

    data = get_historical_daily(station_id, query_date, SERVICE_KEY)
    # print(f"{query_date} ASOS 일별 관측:")
    # for k, v in data.items():
    #     print(f"  {k}: {v}")

    weatherscore=weatherScore.calculate_weather_score(float(data['T']), float(data['RH']), float(data['P']), float(data['W']), float(data['S']))
    return weatherscore

if __name__ == '__main__':
    #예시: 2025년 5월 12일, 서울(108) 자료
    SERVICE_KEY = 'ovbSd4fqkM8YCBsJZb3IXRVetj2J5nfwIfpFULsUyXn7nirAutYIrpAaLxFIG6bHP3mbSp203cz1k385t8AbYQ=='
    query_date = '20231213'    # YYYYMMDD
    weatherscore = weather_score(query_date)
    print(f"{query_date} ASOS 일별 관측:")
    for k, v in get_historical_daily(108, query_date, SERVICE_KEY).items():
        print(f"  {k}: {v}")
    print(f"Weather Amenity Score: {weatherscore} / 100")