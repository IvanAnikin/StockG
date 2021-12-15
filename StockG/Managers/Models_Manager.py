import requests
import json

scoring_uri = 'http://21ed1e07-2233-486a-af57-47fe5585ce4c.westeurope.azurecontainer.io/score'
key = ''

data = {"data":
        [
            [
                0.0199132141783263,
                0.0506801187398187,
                0.104808689473925,
                0.0700725447072635,
                -0.0359677812752396,
                -0.0266789028311707
            ],
            [
                -0.0127796318808497,
                -0.044641636506989,
                0.0606183944448076,
                0.0528581912385822,
                0.0479653430750293,
                0.0293746718291555]
        ]
        }
input_data = json.dumps(data)

headers = {'Content-Type': 'application/json'}
if key!='': headers['Authorization'] = f'Bearer {key}'

resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)