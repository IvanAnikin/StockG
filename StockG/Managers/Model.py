import requests
import json
import yfinance as yf
import pandas as pd

class Model():
    def __init__(self, args):
        self.args = args
        if(args['url'!='']):
            allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.
        #elif(args['path'!='']):

    def predict(self, input):
        if(self.args['url'!='']):

            #data = data.values.tolist()
            # Request data goes here
            # body={"data": data[0]}
            # body={data}
            # body = str.encode(data.to_json())#json.dumps(data)

            body = {"data": input}

            url = self.args['url']
            api_key = ''  # Replace this with the API key for the web service
            headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

            req = urllib.request.Request(url, body, headers)

            try:
                response = urllib.request.urlopen(req)

                result = response.read()
            except urllib.error.HTTPError as error:
                print("The request failed with status code: " + str(error.code))

                # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
                print(error.info())
                print(json.loads(error.read().decode("utf8", 'ignore')))

            # 2nd method
            #key=''
            #body = {"data": data}

            #headers = {'Content-Type': 'application/json'}
            #if key != '': headers['Authorization'] = f'Bearer {key}'

            #resp = requests.post(self.args['url'], body, headers=headers)

        #elif (args['path' != '']):
            # result = loadmodel(path).predict

        return result

    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context',
                                                                               None):
            ssl._create_default_https_context = ssl._create_unverified_context

