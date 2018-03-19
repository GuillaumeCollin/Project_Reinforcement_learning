# Utilisation de OpenAI Gym pour avoir l'agent

import gym
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, SeparableConv2D, Flatten, Input
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
import pandas as pd
import numpy as np
import sys
import random
from skimage.transform import rescale, resize
from scipy.misc import toimage
import pickle
print(sys.path)


env = gym.make('CartPole-v0')
state_size = env.observation_space.shape
action_size = env.action_space.n
Q_value = []
class DQN_Agent():
    def __init__(self, state_size, action_size):
        self.gamma = 0.99
        self.lr = 0.001
        self.memory = []

        self.aggregated_steps = 4
        self.state_size = state_size
        self.action_size = action_size
        
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.decay = 0.9
        
        
    def initNetwork(self, filename='None'):
        if filename=='None':
            self.model = self.neural_net()
        else:
            print('load : ', filename)
            self.model = load_model(filename) 
                 
    def neural_net(self):
        # Neural Net for Deep Q Learning
        # Sequential() creates the foundation of the layers.
        model = Sequential()
        model.add(Dense(32, input_shape=self.state_size))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
        return model
    
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
        # reduction epislon ?
        #print(next_actions.shape)
        #print(next_actions)
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
        actions = np.array(actions, dtype=np.int8)
        #print(actions)
        
        target = self.model.predict(states)
        
        #print(states.shape)
        #print(actions.shape)
        #print(target.shape)
        for i in range(batch_size):
            target[i, actions[i]] = y[i]
        #print(target)
        #print(y)
        self.model.train_on_batch(states, target)
            
        #exit()
    def test(self):     
        time_t = 0
        score = 0
        terminal = False
        repeat_action = 1
        # reset state 
        state = env.reset()
        #state = agent.resize(agent.RGBtoGray(state))
        
        history = []
        for i in range(0, repeat_action):
            history.append(state)

        state_memory = np.array(history, dtype=np.float16).reshape(agent.state_size)
        #print("state memory shape : ", state_memory.shape)       
        
   
        while(terminal == False and time_t <= 2000):
            #env.render()
            #state_memory = np.array(history, dtype=np.float16).reshape((agent.image_height, agent.image_width, agent.aggregated_images))
            
            action = self.act(np.expand_dims(state_memory, axis=0))

            
            for i in range(repeat_action):
                next_state, reward, terminal, _ = env.step(action)
                score += reward

                # next_state = agent.resize(agent.RGBtoGray(next_state))
                history.pop(0)
                history.append(next_state)
                
                if terminal:
                    print("episode: {}/{}, memory: {}, step: {},  score: {}, epsilon: {}".format(e, nb_episodes, len(agent.memory), time_t, score, agent.epsilon)) 
                    break


            if len(self.memory) == N:
               self.memory.pop(0)
            #state_memory = np.array(history, dtype=np.float16).reshape((agent.image_height, agent.image_width, agent.aggregated_images))
            next_state_memory = np.array(history, dtype=np.float16).reshape(agent.state_size)
            #print("next state memory shape : ", next_state_memory.shape)                
                       
#            agent.add_memory(state=state_memory, action=action, reward=self.normalize_reward(reward), next_state=next_state_memory, terminal=terminal)    
                
                
            state_memory = next_state_memory
            
            time_t += 1
        return score, time_t 
        
if __name__ == "__main__":


    filename = 'DQN_cartpolev2.hdf5'
    print('size : ', action_size)
    agent = DQN_Agent(state_size, action_size)
    agent.initNetwork()
    env._max_episode_steps = None
    nb_episodes = 1000000
    batch_size = 32
    repeat_action = 1
    best_score = 0
    N= 1000000 #memory size
    # Iterate the game
    #f = open('average_rewards_cartpole.txt', 'w')
    mean_score = 0
    array_score = []
    frame_t = 0
    for e in range(0, nb_episodes):
        # initialisation episode
        time_t = 0
        score = 0
        terminal = False

        # reset state 
        state = env.reset()

        
        history = []
        for i in range(0, repeat_action):
            history.append(state)

        state_memory = np.array(history, dtype=np.float16).reshape(agent.state_size)
        #print("state memory shape : ", state_memory.shape)       
        
   
        while(terminal == False and time_t <= 2000):
            
            
            action = agent.act(np.expand_dims(state_memory, axis=0))

            
            for i in range(repeat_action):
                next_state, reward, terminal, _ = env.step(action)
                score += reward

                # next_state = agent.resize(agent.RGBtoGray(next_state))
                history.pop(0)
                history.append(next_state)
                
                if terminal or e == nb_episodes-1:
                    print("episode: {}/{}, memory: {}, step: {}, iteration: {}, score: {}, epsilon: {}".format(e, nb_episodes, len(agent.memory), time_t, frame_t, score, agent.epsilon)) 
                    break


            if len(agent.memory) == N:
               agent.memory.pop(0)
            #state_memory = np.array(history, dtype=np.float16).reshape((agent.image_height, agent.image_width, agent.aggregated_images))
            next_state_memory = np.array(history, dtype=np.float16).reshape(agent.state_size)
            #print("next state memory shape : ", next_state_memory.shape)                
                       
            agent.add_memory(state=state_memory, action=action, reward=agent.normalize_reward(reward), next_state=next_state_memory, terminal=terminal)    
                
                
            state_memory = next_state_memory
            
            time_t += 1
            frame_t +=1


            if len(agent.memory) > batch_size and frame_t%1000 == 0:
                agent.replay(batch_size)
                agent.model.save(filename)
 #               print('model save')
            elif len(agent.memory) > batch_size:
                agent.replay(batch_size)
#                exit()
#                agent.model.save('DQN_keras.hdf5')
            
            if frame_t < 1000000:
		        agent.epsilon = agent.epsilon-agent.decay/1000000
            else:
                agent.epsilon = 0.1
        del history        
     
        if e % 100 == 0:
                mean_score += score
                array_score.append(mean_score)
                mean_score = 0
                pickle.dump( array_score, open( "score_average_cartpolev2.pkl", "wb" ) )        
                score_test, _ = agent.test()
                if best_score < score_test:
                    agent.model.save(filename)
                    print('model save, score : ', score_test, ' , best score ', best_score)
                    best_score = score_test
        else:
                mean_score += score

            
    
