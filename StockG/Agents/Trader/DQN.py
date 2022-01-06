
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input, concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import math
import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time
import io

class DQN():
    def __init__(self, args):
        self.args=args

        self.model = self.model_builder()
        self.random_moves = 0
        self.non_random_moves = 0

        model_full_path = "{}\{}\{}.h5".format(os.getcwd(), args['models_dir'], args['model_name'])
        if args['load_model']:
            if os.path.exists(model_full_path):
                self.model.load_weights(model_full_path)
                print("Weights loaded | {path}".format(path=model_full_path))
            else: print(f"Couldn't find any model with path '{model_full_path}'")

    def model_builder(self):

        inputs = Input(shape=self.args['state_size'])

        for (i, f) in enumerate(self.args['filters']):

            if i == 0:
                x = inputs
            x = Dense(f)(x)
            x = Activation("relu")(x)
            #if (self.filters_dropouts[i] != 0): x = Dropout(self.dropouts[i])(x)
            #x = BatchNormalization(axis=chanDim)(x)

        x = Dense(self.args['action_space'])(x)
        x = Activation(self.args['activation'])(x)

        model = Model(inputs, x)
        model.compile(loss='mse', optimizer=self.args['optimizer'])
        return model

    def trade(self, state):
        if random.random() <= self.args['epsilon'] or state.size<10:
            self.random_moves += 1
            return random.randrange(self.args['action_space'])
        self.non_random_moves += 1
        return self.model.predict(state)

    def batch_train(self, batch_size):

        batch = []
        for i in range(len(self.args['memory']) - batch_size + 1, len(self.args['memory'])):
            batch.append(self.args['memory'][i])

        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.args['gamma'] * np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            if(int(action)>2):
                print(action)
            target[0][int(action)] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.args['epsilon'] > self.args['epsilon_final']:
            self.args['epsilon'] *= self.args['epsilon_decay']
            #print(self.args['epsilon'])

    def sigmoid(self, x):
        return 1/(1 + math.exp(-x))

    def stock_price_format(self, n):
        if n < 0:
            return "- # {0:2f}".format(abs(n))
        else:
            return "$ {0:2f}".format(abs(n))

    def state_creator(self, data, timestep, window_size):

        starting_id = timestep - window_size + 1

        if starting_id >= 0:
            windowed_data = data[starting_id:timestep + 1]
        else:
            windowed_data = starting_id * [data[0]] + list(data[0:timestep + 1])

        state = []
        for i in range(window_size - 1):
            if(len(windowed_data)>i+1):
                state.append(self.sigmoid(windowed_data[i + 1] - windowed_data[i]))

        return np.array([state])

    def train(self, data):

        total_profits = []
        trading_memory = [] # [data, sold, price]
        writer = tf.summary.create_file_writer(logdir=self.args['tensorboard_log_dir'])

        for episode in range(1, self.args['episodes'] + 1):

            state = self.state_creator(data, episode, self.args['window_size'] + 1)

            total_profit = 0
            self.inventory = []

            episode_start = time.time()
            #for t in tqdm(range(self.args['data_samples'])):
            for t in range(self.args['data_samples']):

                action = np.argmax(self.trade(state))

                next_state = self.state_creator(data, t + 1, self.args['window_size'] + 1)
                reward = 0

                if action == 1:  # Buying
                    self.inventory.append(data[t])
                    trading_memory.append([data.keys()[t], 0, data[t]])

                elif action == 2 and len(self.inventory) > 0:  # Selling
                    buy_price = self.inventory.pop(0)

                    reward = max(data[t] - buy_price, 0)
                    total_profit += data[t] - buy_price
                    trading_memory.append([data.keys()[t], 1, data[t]])
                    #self.args['trading_memory'].append(data)

                if t == self.args['data_samples'] - 1:
                    done = True
                else:
                    done = False

                if(state.size==self.args['window_size']):
                    self.args['memory'].append((state, action, reward, next_state, done))

                state = next_state
                if len(self.args['memory']) > self.args['batch_size'] and t%self.args['train_every']==0:
                    self.batch_train(self.args['batch_size'])

                # Visualisation
                if self.args['visualise'] and t%self.args['train_every']==0 and t!=0:
                    sys.stdout.write(
                        "\r Timestep: {timestep}/{timesteps} | Time: {time_spent}/{time_total} | Random moves: {random_moves} | Non random moves {non_random_moves} | Epsilon: {epsilon}".format(
                            timestep=t, timesteps=self.args['data_samples'], time_spent = time.strftime('%H:%M:%S', time.gmtime(time.time()-episode_start)),
                            time_total= time.strftime('%H:%M:%S', time.gmtime((time.time()-episode_start)/t*self.args['data_samples'])),
                            random_moves=self.random_moves, non_random_moves=self.non_random_moves, epsilon=np.round(self.args['epsilon'],2)))
                    sys.stdout.flush()
                    if (t % 200 == 0): print()

            total_profits.append(total_profit)
            tf.summary.scalar(name="reward", data=total_profit, step=episode)
            dqn_variable = self.model.trainable_variables
            tf.summary.histogram(name="dqn_variables", data=tf.convert_to_tensor(dqn_variable[0]), step=episode)
            writer.flush()

            # Visualisation
            if self.args['visualise']:
                if(episode>50): sys.stdout.write("\r Episode: {episode}/{episodes} | Avg total Profit: {profit}".format(
                    episode=episode, episodes=self.args['episodes'], profit=np.average(total_profits[episode-50:episode])))
                else: sys.stdout.write("\r Episode: {episode}/{episodes}".format(
                    episode=episode, episodes=self.args['episodes']))
                sys.stdout.flush()
                print()
            #Saving model
            if episode % 10 == 0:
                self.model.save("{dir}/{name}.h5".format(dir=self.args['models_dir'], name=self.args['model_name']))

        np.save("{dir}/total_profits.npy".format(dir=self.args['logs_dir']), total_profits)
        np.save("{dir}/trading_memory.npy".format(dir=self.args['logs_dir']), trading_memory)

    def visualise_trading(self, window = 20):

        dir = "{}\{}\{}".format(os.getcwd(), self.args['logs_dir'], "trading_memory.npy")
        if os.path.exists(dir):
            trading_memory = np.load(dir, allow_pickle=True)
            total_profits = np.load("{}\{}\{}".format(os.getcwd(), self.args['logs_dir'], "total_profits.npy"), allow_pickle=True)

            episodes = []
            for i in range(len(total_profits)):
                if (i+1)%window==0: episodes.append(i)
            total_profits_averages = np.average(total_profits.reshape(-1, window), axis=1)

            selling = []
            buying = []
            for step in trading_memory:
                if(step[1]): selling.append([step[0], step[2]])
                else: buying.append([step[0], step[2]])
            print("selling avg.: {} | buying avg.: {}".format(np.average(np.array(selling)[:, 1]), np.average(np.array(buying)[:, 1])))

            fig, ax = plt.subplots(figsize=(16, 9))
            ax.plot(episodes, total_profits_averages, label='Profit development')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Avg. profit')
            ax.legend()
            plt.show()

            # TENSORBOARD PLOTTING - https://stackoverflow.com/questions/38543850/how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots
            #buf = io.BytesIO()
            #plt.savefig(buf, format='png')
            #buf.seek(0)
            ## Prepare the plot
            #plot_buf = gen_plot()

            ## Convert PNG buffer to TF image
            #image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

            ## Add the batch dimension
            #image = tf.expand_dims(image, 0)

            ## Add image summary
            #tf.summary.image("plot", image)
            #writer = tf.summary.create_file_writer('../Train/logs')
            #writer.flush()

            # Session
            #with tf.Session() as sess:
                # Run
                #summary = sess.run(summary_op)
                # Write summary
                #writer = tf.train.SummaryWriter('./logs')
                #writer.add_summary(summary)
                #writer.close()

            #ax.plot(np.array(selling)[:, 0], np.array(selling)[:, 1], color="green")
            #ax.plot(np.array(buying)[:, 0], np.array(buying)[:, 1], color="red")

        else: print(f"Couldn't find trading data in '{dir}'")