import requests

url = 'http://localhost:9696/predict'
#url = 'https://marvel-216656375128.europe-north1.run.app/predict'

character = {
    'id' : 'Secret Identity',
    'ilign' : 'Good Characters',
    'eye' : 'Green Eyes',
    'hair' : 'Red Hair',
    'alive' : 'Livivng Characters',
    'appearances' : '5',
    'first_appearances' : '1947-07-05'
}

response = requests.post(url, json=character).json()
print(response)