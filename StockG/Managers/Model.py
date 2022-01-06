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
            converted_input['Date'] = input['Date']
            for key, value in input.items():
                if(key!='Adj Close' and key!='Date'): converted_input[key] = float(value)
            json_data = dict(converted_input)
            body = {
                "data": [json_data],
                  "quantiles": [
                    self.args['quantile1'],#0.025,
                    self.args['quantile2']#0.975
                  ]}
            body_json = json.dumps(body)

            url = self.args['url']

            #headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}
            headers = {'Content-Type': 'application/json; charset=utf-8'}
            api_key = ''  # Replace this with the API key for the web service
            if api_key != '': headers['Authorization'] = f'Bearer {key}'


            result = requests.post(url, body_json, headers=headers)

            # IF RESULT NOT 200

        #elif (args['path' != '']):
            # result = loadmodel(path).predict

        return result

    def allowSelfSignedHttps(self, allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context',
                                                                               None):
            ssl._create_default_https_context = ssl._create_unverified_context

