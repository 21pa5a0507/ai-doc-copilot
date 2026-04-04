import requests
import time

url = "https://www.hexnode.com/mobile-device-management/help"
for i in range(200):
    response = requests.get(url)
    print(i, response.status_code)
    time.sleep(0.1)