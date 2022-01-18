import time
import requests
url = 'https://europe-west1-fleet-reserve-338511.cloudfunctions.net/function-1'
payload = {'message': 'Hello, General Kenobi'}

for _ in range(1000):
   r = requests.get(url, params=payload)
   print("sent message")




