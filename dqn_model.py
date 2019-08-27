import numpy as np
from keras.models import load_model
import random
from collections import deque
from rl import core
from keras.layers import Dense, Conv1D, Reshape, Add
from keras.models import Sequential, Model
import keras.backend.tensorflow_backend as K
from keras.optimizers import Adam
import os


# Downsample (average) data
def state_input(state):
    ts = state[:, 2].reshape([10, 5]).mean(1)
    a5 = state[:, 0].reshape([10, 5]).mean(1)
    a6 = state[:, 1].reshape([10, 5]).mean(1)
    return ts, a5, a6


# Basic DQN Agent Class with simple residual convolution layer
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=(12000))
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_min = 0
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.model = self._build_model()


    def _build_model(self):                                                         
        with K.tf.device('/gpu:0'):
            # CNN + Residual
            conv = Sequential()
            conv.add(Reshape((10, 1), input_shape=(10,)))
            conv.add(Conv1D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal',
                            kernel_regularizer='l2'))
            conv.add(Conv1D(1, 1, activation='relu', kernel_initializer='he_normal', kernel_regularizer='l2'))
            conv.add(Reshape((10,)))

            res = Add()([conv.input, conv.output])

            fc = Reshape((10,))(res)

            fc = Dense(64, activation='relu', kernel_initializer='he_normal',
                       kernel_regularizer='l2')(fc)
            fc = Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer='l2')(fc)
            fc = Dense(5, activation='linear')(fc)

            model = Model(inputs=[conv.input], outputs=[fc])
            model.compile(loss='mse',
                          optimizer=Adam(lr=self.learning_rate))

            ''' FC Network
            model = Sequential()
            model.add(Dense(64, activation='relu', kernel_initializer='he_normal',
                       kernel_regularizer='l2', input_shape=(10,)))
            model.add(Dense(64, activation='relu', kernel_initializer='he_normal',
                            kernel_regularizer='l2'))
            model.add(Dense(64, activation='relu', kernel_initializer='he_normal',
                            kernel_regularizer='l2'))
            model.add(Dense(5, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
            '''
            model.summary()

            return model

    def remember(self, state, action, reward, next_state, done):                   
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):                                                          
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):                                                  
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
