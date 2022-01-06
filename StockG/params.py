from datetime import datetime
from tensorflow.keras.optimizers import Adam
from collections import deque
import datetime as dt

expected_mapping = {
    "SMA": "Close",
    "SMA_Volume": "Volume",
    "ATR": "TR",
    "ADX": "DX"
}

dataset_args={
    # 'dataset_path': "",
    # 'dataset_url': "",
    'store_path': "",
    'dir': "../Datasets",
    'dataset_info':{
        'name': 'MSFT',
        'start': dt.datetime(2021, 1, 1),
        'end': dt.date.today(),
        'progress': False
    },
    'X':{
        'fundamental_data': {

        },
        'windowed_data': [
            "SMA",
            "SMA_Volume",
            "ATR",
            "ADX"
            # Average True Range
            # Average Directional Index (Fast and Slow)
            # Stochastic Oscillators (Fast and Slow)
            # Relative Strength Index (Fast and Slow)
            # Moving Average Convergence Divergence
            # Bollinger Bands
            # Rate of Change
        ]
    },
    'Y': 'Adj Close',
    'n': 10,
    'r': 10
}
general_args={
    "brand_name" : "StockG",
    "year": datetime.now().year
}
model_args={
    'url':'http://21ed1e07-2233-486a-af57-47fe5585ce4c.westeurope.azurecontainer.io/score',
    'quantile1':0.025,
    'quantile2':0.975,
}


default_dqn_args={
    'state_size': 10,
    'action_space': 3,
    'memory': deque(maxlen=2000),
    'inventory': [],
    'model_name': 'DQN',
    'load_model': False,

    'gamma': 0.95,
    'epsilon':1.0,
    'epsilon_final': 0.01,
    'epsilon_decay': 0.995,
    'train_every': 100,

    'filters': [32, 64, 128],
    'activation': 'linear',
    'optimizer': Adam(learning_rate=0.001), #lr

    'window_size': 10,
    'episodes': 1000,

    'batch_size': 32,
    'data_samples': None,
    'models_dir': '../Train/Models',
    'logs_dir': '../Train/Memory/DQN',
    'tensorboard_log_dir': '../Train/logs/DQN',

    'visualise': True,
}

default_nn_args={
    'features': ["adjclose", "volume", "open", "high", "low"],
    'window_size': 50,
    'model_name': 'NN',
    'load_model': True,
    'ticker': 'MSFT',
    'test_size': 0.2,

    'filters': [32, 64, 128],
    'dropout': 0.3,
    'bidirectional': False,
    'activation': 'linear',
    'optimizer': Adam(learning_rate=0.001), #lr
    'loss':'mean_absolute_error',

    'batch_size': 64,
    'epochs': 500,
    'data_samples': None,
    'models_dir': 'Models',
}