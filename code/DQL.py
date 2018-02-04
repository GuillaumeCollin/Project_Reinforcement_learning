from keras.models import Sequential
from keras.layers import Dense,Conv2D,Activation, Flatten
from keras.optimizers import Adam
import numpy as np
import random
from keras.models import load_model


# Deep Q-learning Agent

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.image_height = 84
        self.image_width = 84
        self.aggregated_images = 4
        self.memory = []
        self.learning_rate = 0.001
        self.action_size = action_size
        self.state_size = state_size
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.95
        self.model = self._build_model()
        self.max_size_memory = 100000

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8,8), strides=(4,4), input_shape=(self.image_height, self.image_width, self.aggregated_images)))
        model.add(Activation('relu'))
        model.add(Conv2D(32,(4,4), strides=(2,2)))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def continue_model(self,file):
        path = 'output/' + file + '.h5'
        model = load_model(path)
        self.model = model

    def add_to_memory(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        if len(self.memory) > self.max_size_memory:
            self.memory.remove(self.memory[0])

    def act(self, state):
        #Greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        next_action = self.model.predict(state)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.argmax(next_action[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory,batch_size)
        for state,action, reward,next_state,terminal in minibatch:
            if terminal:
                y = reward
            else :
                y = reward + self.gamma * np.max(self.model.predict(np.array(next_state)))
            y_cible = self.model.predict(state)
            y_cible[0][action] = y
            self.model.fit(state, y_cible, epochs=1, verbose=0)


    def save(self, path):
        self.model.save(path)