# Utilisation de OpenAI Gym pour avoir l'agent

import gym
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, SeparableConv2D, Flatten, Lambda, Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import load_model
import pandas as pd
import numpy as np
import sys
import random
from skimage.transform import rescale, resize
from scipy.misc import toimage
import pickle
import math
print(sys.path)


env = gym.make('Breakout-v0')
state_size = env.observation_space.shape
action_size = env.action_space.n

def mean(liste):
   return float(sum(liste))/ len(liste)

class DQN_Agent():
    def __init__(self, state_size, action_size):
        self.gamma = 0.99 #discount
        self.lr = 0.01
        self.memory = []
        self.image_height=84
        self.image_width= 84 
        self.aggregated_images = 4
        self.state_size = state_size
        self.action_size = action_size
        
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.decay = 9 * math.pow(10, -7)
        
        
    def initNetwork(self, filename='None'):
        if filename=='None':
            self.model = self.neural_net()
        else:
            print('load filename : ', filename)
            self.model = load_model(filename) 
            
    def RGBtoGray(self, image):
        gray_image = image[:,:, 0]*0.299 + image[:,:, 1]*0.587 + image[:,:, 2]* 0.114
  
        return np.round(gray_image)
    
    def resize(self, image, downsampling=(110, 84), crop=(84, 84)):
        temp = resize(image[0:210,0:160], downsampling)
        #print(temp)
        
        #toimage(temp).show()
        new = temp[20:104, :]
        #print(crop.shape)
        #toimage(crop).show()
        return new.astype(np.uint8) # memory limitation
        
    def neural_net(self):
        # Neural Net for Deep Q Learning
        # Sequential() creates the foundation of the layers.
#        model = Sequential()
#        
#        
#        
#        
#        model.add(Lambda(lambda x: x / 255.0))
#        model.add(SeparableConv2D(16, (8,8), strides=(4,4), input_shape=(self.image_height, self.image_width, self.aggregated_images)))
#        model.add(Activation('relu'))
#        model.add(SeparableConv2D(32, (4,4), strides=(2,2)))
#        model.add(Activation('relu'))
#        model.add(Flatten())
#        model.add(Dense(256))
#        model.add(Activation('relu'))
#        model.add(Dense(self.action_size, activation='linear'))
        
        x = Input(shape=(self.image_height, self.image_width, self.aggregated_images))
        model = Lambda(lambda x: x / 255.0)(x)
        model = SeparableConv2D(16, (8,8), strides=(4,4))(model)
        model = Activation('relu')(model)
        model = SeparableConv2D(32, (4,4), strides=(2,2))(model)
        model = Activation('relu')(model)
        model = Flatten()(model)
        model = Dense(256)(model)
        model = Activation('relu')(model)
        model = Dense(self.action_size, activation='linear')(model)
        mod = keras.models.Model(input=x, output=model)
        mod.compile(loss='mse', optimizer=Adam()) # RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
        return mod
    
    ## Loss = (target - predictions)^2
    def target(self, next_state, reward, discount, terminal):
        if terminal == True:
            return 0
        else:
            return reward + discount * np.amax(self.model.predict(np.expand_dims(next_state,axis=0)))
    
    def prediction(self, current_state):
        return self.model.predict(current_state)
    
    
    def add_memory(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        
    
    
    def act(self, state):
        # epislon greedy
        if np.random.rand() < self.epsilon:
            # action random
            return env.action_space.sample()
            
        next_actions = self.model.predict(state)[0]

        return np.argmax(next_actions)
    
    def normalize_reward(self, reward):
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return 0 
            
    def replay(self, batch_size):
        ## Sample minibatch from the memory
        minibatch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        y = []
        for state, action, reward, next_state, terminal in minibatch:
            states.append(state)
            actions.append(action)
            y.append(self.target(next_state=next_state, reward=reward, discount=self.gamma, terminal = terminal))
            
        states = np.array(states, dtype=np.float16)
        actions = np.array(actions, dtype=np.int16)
        
        target = self.model.predict(states)

        for i in range(batch_size):
            target[i, actions[i]] = y[i]
        #print(target)
        #print(y)
        self.model.train_on_batch(states, target)
            
        #exit()
        return mean(y)
        
    
if __name__ == "__main__":

    agent = DQN_Agent(state_size, action_size)
    filename2 = 'DQN_keras_adam'
    filename = 'DQN_keras_adam2'
#    agent.initNetwork(filename2+'.hdf5')
    agent.initNetwork()
    nb_episodes = 4000000
    batch_size = 32
    repeat_action = 4
    N= 1000000 #memory size


	
    mean_score = 0
    array_score = []
    Qm = []
    Q_average = []
    frame_t = 0
    for e in range(0, nb_episodes):
        # initialisation episode
        time_t = 0
        score = 0
        terminal = False
        
        # reinitialise l'etat
		
        state = env.reset()
        state = agent.resize(agent.RGBtoGray(state))
        
        history = []
        for i in range(0, agent.aggregated_images):
            history.append(state)

        state_memory = np.array(history, dtype=np.uint8).reshape((agent.image_height, agent.image_width, agent.aggregated_images))
        
              
        
        # iterate step = 4 frames
        while(terminal == False):
            
            
            
            action = agent.act(np.expand_dims(state_memory, axis=0))

            
            for i in range(repeat_action):
                next_state, reward, terminal, _ = env.step(action)
                score += reward
                
                next_state = agent.resize(agent.RGBtoGray(next_state))
                history.pop(0)
                history.append(next_state)
                
                if terminal:
                    # affiche le score d'un episode
                    print("episode: {}/{}, memory: {}, step: {}, iteration : {}, score: {}, Q_mean: {}, epsilon: {}".format(e, nb_episodes, len(agent.memory), time_t, frame_t, score, mean(Qm), agent.epsilon))
                    break


            if len(agent.memory) == N:
               
               agent.memory.pop(0)
            #state_memory = np.array(history, dtype=np.float16).reshape((agent.image_height, agent.image_width, agent.aggregated_images))
            next_state_memory = np.array(history, dtype=np.uint8).reshape((agent.image_height, agent.image_width, agent.aggregated_images))
                
            agent.add_memory(state=state_memory, action=action, reward=agent.normalize_reward(reward), next_state=next_state_memory, terminal=terminal)    
                
                
            state_memory = next_state_memory
            
            time_t += 1
            frame_t += 1

            # train and save the model
            if frame_t%1000 == 0 and len(agent.memory)>batch_size:
                Qm.append(agent.replay(batch_size))
                agent.model.save(filename+'.hdf5')
            elif len(agent.memory)>batch_size:
                Qm.append(agent.replay(batch_size))
#                agent.model.save('DQN_keras.hdf5')
                
			# change epsilon
            if frame_t <= 1000000:
                agent.epsilon = agent.epsilon-agent.decay
            else:
                agent.epsilon = agent.epsilon_min
        del history        
     
        # save score and Q value (average)        
        if e % 500 == 0:
                mean_score += score
                array_score.append(mean_score)
                mean_score = 0
                Q_average.append(mean(Qm))
                Qm = []
                pickle.dump( array_score, open( "score_average_"+filename+".pkl", "wb" ) )        
                pickle.dump( Q_average, open( "Q_average_"+filename+".pkl", "wb" ) )
        else:
                mean_score += score
        
    
