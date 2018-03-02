from keras.models import Sequential
from keras.layers import Dense,Conv2D,Activation, Flatten
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
import random
from keras.models import load_model

# Deep Q-learning Agent

class DQLAgent:
    def __init__(self, state_size, action_size , nb_episodes_total):
        self.image_height = 84
        self.image_width = 84
        self.aggregated_images = 4
        self.memory = []
        self.learning_rate = 0.001
        self.action_size = action_size
        self.state_size = state_size
        self.epsilon = 1
        self.nb_episodes_total = nb_episodes_total
        self.epsilon_decay = 1/nb_episodes_total
        self.epsilon_min = 0.1
        self.gamma = 0.95
        self.model = self._build_model()
        self.max_size_memory = 1000000

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8,8), strides=(4,4), input_shape=(self.image_height, self.image_width, self.aggregated_images)))
        model.add(Activation('relu'))
        model.add(Conv2D(32,(4,4), strides=(2,2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=self.learning_rate))
        return model

    def continue_model(self,file):
        path = 'output/' + file + '.h5'
        model = load_model(path)
        print('Model successfully loaded')
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
            self.epsilon -= self.epsilon_decay
        return np.argmax(next_action)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory,batch_size)
        states = np.zeros((batch_size, self.image_height,self.image_width,self.aggregated_images))
        y_cible = np.zeros((batch_size, self.action_size))

        for ligne in range(len(minibatch)):
            state = minibatch[ligne][0]
            states[ligne] = state
            action = minibatch[ligne][1]
            reward = minibatch[ligne][2]
            next_state = minibatch[ligne][3]
            terminal = minibatch[ligne][4]
            if terminal:
                y = reward
            else:
                y = reward + self.gamma * np.max(self.model.predict(np.array(next_state)))
            y_cible[ligne] = self.model.predict(state)
            y_cible[ligne][action] = y
        self.model.train_on_batch(states, y_cible)

    def save(self, path):
        self.model.save(path)

    def update_epsilon(self,nb_episodes):
        print(('Reprise du model au {} eme episode').format(nb_episodes))
        self.epsilon = 1 - nb_episodes/self.nb_episodes_total