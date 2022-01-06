"""
Routes and views for the flask application.
"""

import json
from flask import render_template, request, send_file
from StockG import app

from StockG.Managers import Dataset_Manager
from StockG import params
from StockG.Agents.Trader import technical

from PIL import Image as im
import io


dataset_manager = Dataset_Manager.Datasets_Manager(args=params.dataset_args)
brand_name = params.general_args['brand_name']
year = params.general_args['year']


# Intro Page
@app.route("/intro")
def intro():
    return render_template(
        'intro.html',
        title='Intro',
        year=year,
        brand_name = brand_name
    )


# StockG UI
@app.route('/')
@app.route('/home')
def home():
    return render_template(
        'index.html',
        title='Home',
        year=year,
        brand_name = brand_name
    )

# Requst hanglers 

@app.route('/load_dataset', methods=['GET', 'POST'])
def load_dataset():

    dataset_info = {'name':request.args.get('name')}
    dataset_manager.args={'dataset_info': dataset_info}
    dataset_manager.load_dataset()
    dataset = dataset_manager.convert_to_json(dataset_manager.dataset)
    try:
        return json.dumps(dataset)
    except Exception as e:
        return e

@app.route('/load_technical_signals', methods=['GET', 'POST'])
def load_technical_signals():

    log_string, signals_img, portfolio_img = technical.get_signals(stock_name=request.args.get('name'))

    signals_img.seek(0)

    try:
        return send_file(signals_img, mimetype='image/PNG')
    except Exception as e:
        return e

@app.route('/technical_signals', methods=['GET', 'POST'])
def technical_signals():

    #log_string, signals_img, portfolio_img = technical.get_signals(stock_name=request.args.get('name'))
    best, log_string, Return, signals_img, portfolio_img = technical.get_signals_loop()

    #signals_img.seek(0)

    try:
        return ":)"
        #return send_file(signals_img, mimetype='image/PNG')
    except Exception as e:
        return e
