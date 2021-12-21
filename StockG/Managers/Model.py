import requests
import json
import yfinance as yf
import pandas as pd
import urllib
import os
import ssl
from collections import OrderedDict

class Model():
    def __init__(self, args):
        self.args = args
        if(args['url']!=''):
            self.allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.
        #elif(args['path'!='']):

    def predict(self, input):
        if(self.args['url']!=''):
            converted_input = OrderedDict()
            for key, value in input.items():
                if(key!='Date'): converted_input[key] = float(value)
                elif(key!='Adj Close'): converted_input[key] = value
            json_data = dict(converted_input)
            body = {"data": [json_data],
              "quantiles": [
                self.args['quantile1'],#0.025,
                self.args['quantile1']#0.975
              ]}
            print(body)
            body = {
              "data": [
                {
                  "Date": "2021-01-01T00:00:00.000Z",
                  "Open": 0,
                  "High": 0,
                  "Low": 0,
                  "Close": 0,
                  "Volume": 0
                }
              ],
              "quantiles": [
                0.025,
                0.975
              ]
            }
            body = json.dumps(body)
            #body = bytes(body, 'utf-8')

            url = self.args['url']

            #headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
            headers = {'Content-Type': 'application/json; charset=utf-8'}
            #headers = {}
            api_key = ''  # Replace this with the API key for the web service
            if api_key != '': headers['Authorization'] = f'Bearer {key}'


            result = requests.post(url, body, headers=headers)

            #req = urllib.request.Request(url, body, headers)
            #try:
            #    response = urllib.request.urlopen(req)
            #    result = response.read()
            #except urllib.error.HTTPError as error:
            #    print("The request failed with status code: " + str(error.code))
            #    print(error.info())
            #    print(json.loads(error.read().decode("utf8", 'ignore')))

            print(result)
        #elif (args['path' != '']):
            # result = loadmodel(path).predict

        return result

    def allowSelfSignedHttps(self, allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context',
                                                                               None):
            ssl._create_default_https_context = ssl._create_unverified_context

