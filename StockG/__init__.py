"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__, subdomain_matching=True)

import StockG.views
