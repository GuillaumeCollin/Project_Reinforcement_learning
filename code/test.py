from keras.models import load_model
import numpy as np
import gym
from preprocess import preprocess
import time
import random


def test(file, render, nb_iter, epsilon,action_size):

    env = gym.make('Breakout-v0')
    state = env.reset()
    path = 'output/' + file + '.h5'
    model = load_model(path)
    average_reward = 0
    q_average = 0
    repeat_action = 4
    history=[0,0,0,0]
    history[0] = state
    history[1] = state
    history[2] = state
    history[3] = state
    prepross_history = preprocess(history)
    compteur = 0
    done = False

    for iter in range(nb_iter):
        while not done:
            compteur += 1
            if render:
                env.render()
                time.sleep(0.1)

            # on cherche la meilleure action
            if np.random.rand() <= epsilon:
                action_prediction = random.randrange(action_size)
            else:
                action_prediction = model.predict(prepross_history)
            q_average += np.max(action_prediction)
            action = np.argmax(action_prediction)

            # On la fait
            for do_action in range(repeat_action):
                next_state_not_processed, reward, done, info = env.step(action)
                history[0:3] = history[1:4]
                history[3] = next_state_not_processed
                next_state = preprocess(history)
                average_reward += reward
            prepross_history = next_state
            # On la sauvegarde

    return average_reward/nb_iter, q_average/nb_iter

def act(state,model):

    next_action = model.predict(state)

    return np.argmax(next_action[0])  # returns action

def main():
    test('first_model.h5',1000)
