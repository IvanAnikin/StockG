#from os import environ
#from StockG import app
#
#if __name__ == '__main__':
#   HOST = environ.get('SERVER_HOST', 'localhost')
#   try:
#       PORT = int(environ.get('SERVER_PORT', '5555'))
#   except ValueError:
#       PORT = 5555
#   app.run(HOST, PORT)
#   #app.run()#debug=True

#from StockG import app
#app.run()



from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Stock G"
