"""
Routes and views for the flask application.
"""

import json
from flask import render_template, request
from StockG import app

from StockG.Managers import Dataset_Manager
from StockG import params

dataset_manager = Dataset_Manager.Datasets_Manager(args=params.default_dataset_args)
brand_name = params.general_args['brand_name']
year = params.general_args['year']


# Intro Page
@app.route('/')
@app.route("/intro")
def intro():
    return render_template(
        'intro.html',
        title='Intro',
        year=year,
        brand_name = brand_name
    )


# StockG UI

@app.route('/home')
def home():
    return render_template(
        'index.html',
        title='Home',
        year=year,
        brand_name = brand_name
    )

@app.route('/contact')
def contact():
    return render_template(
        'contact.html',
        title='Contact',
        year=year,
        brand_name = brand_name
        #message='Contacts'
    )

@app.route('/about')
def about():
    return render_template(
        'about.html',
        title='About',
        year=year,
        brand_name = brand_name
        #message='Contacts'
    )


# Requst hanglers 

@app.route('/load_dataset', methods=['GET', 'POST'])
def load_dataset():

    dataset_info = {'name':request.args.get('name')}
    
    dataset = dataset_manager.convert_to_json(dataset_manager.load_dataset(dataset_info))
    try:
        return json.dumps(dataset)
    except Exception as e:
        return e

@app.route('/load_company_info', methods=['GET', 'POST'])
def load_company_info():    #!!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!!

    company_info = dataset_manager.load_company_info(name=name)
    try:
        return json.dumps(dataset)
    except Exception as e:
        return e

