import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'rate':5})

print(r.json())
