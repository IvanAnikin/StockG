
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os

from StockG.Managers.Dataset_Manager import Datasets_Manager
from StockG.Agents.DQN import DQN
from StockG.Agents.NN import NN
from StockG import params

datasets_Manager = Datasets_Manager(args=params.default_dataset_args)
#datasets_Manager.visualise_dataset_close()

dqn_args = params.default_dqn_args
dqn_args['data_samples'] = len(datasets_Manager.dataset)

agent = DQN(args=dqn_args)
agent.train(data=datasets_Manager.dataset['Close'])
#agent.visualise_trading()


#NN_Agent = NN(params.default_nn_args)
#
## create these folders if they does not exist
#if not os.path.isdir("results"):
#    os.mkdir("results")
#
#if not os.path.isdir("logs"):
#    os.mkdir("logs")
#
#if not os.path.isdir("data"):
#    os.mkdir("data")
#
## load the data
#data = NN_Agent.load_data()
## save the dataframe
#data["df"].to_csv("{}ticker_{}".format(params.default_nn_args['models_dir'], params.default_nn_args['ticker']))
#
#
## some tensorflow callbacks
#checkpointer = ModelCheckpoint(os.path.join("results", params.default_nn_args['model_name'] + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
#tensorboard = TensorBoard(log_dir=os.path.join("logs", params.default_nn_args['model_name']))
#
## train the model and save the weights whenever we see
#history = NN_Agent.model.fit(data["X_train"], data["y_train"],
#                    batch_size=params.default_nn_args['batch_size'],
#                    epochs=params.default_nn_args['epochs'],
#                    validation_data=(data["X_test"], data["y_test"]),
#                    callbacks=[checkpointer, tensorboard],
#                    verbose=1)