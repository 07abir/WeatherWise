#main purpose of this file is to convert api to datset 


import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive?latitude=22.5626&longitude=88.363&start_date=2016-11-12&end_date=2026-11-11&daily=weather_code,rain_sum,temperature_2m_max,temperature_2m_min,temperature_2m_mean,wind_gusts_10m_max,winddirection_10m_dominant,sunshine_duration,wind_speed_10m_max,cloud_cover_min,relative_humidity_2m_mean,apparent_temperature_mean,precipitation_sum,dew_point_2m_mean&timezone=auto"



params = {
    "latitude": 22.5726,
    "longitude": 88.3639,
    "start_date": "2016-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "cloud_cover_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "sunshine_duration",
        "pressure_msl_mean"
    ],
    "timezone": "Asia/Kolkata"
}
response = requests.get(url,params=params,timeout=30)

response.raise_for_status()
data = response.json()
print("Response type:",type(data))

if isinstance(data,list):
    data=data[0]

if "error" in data:
     raise ValueError(data.get("reason","Api error"))

daily_data = data["daily"]
df = pd.DataFrame(daily_data)     
df.to_csv("weather_data.csv", index=False)
print("Data collected successfully:")
print(df.head())
print(df.shape)
print("columns:",df.columns.tolist())


