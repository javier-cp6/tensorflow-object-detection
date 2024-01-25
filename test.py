import requests

url = "http://localhost:9696/predict"

data = {
    "url": "https://ichef.bbci.co.uk/news/976/cpsprodpb/1622F/production/_96717609_mediaitem96715861.jpg"
}

result = requests.post(url, json=data).json()
print(result)
