from keras.models import load_model
import numpy as np
import gym
from preprocess import preprocess
import time

def test(file, render):
    env = gym.make('Breakout-v0')
    state = env.reset()
    path = 'output/' + file
    model = load_model(path)
    history=[0,0,0,0]
    history[0] = state
    history[1] = state
    history[2] = state
    history[3] = state
    prepross_history = preprocess(history)
    compteur = 0
    done = False

    while not done:
        compteur += 1
        if render :
            env.render()
        time.sleep(0.1)

        # on cherche la meilleure action

        action = act(prepross_history,model)


        # On la fait
        next_state_not_processed, reward, done, info = env.step(action)

        history[0:3] = history[1:4]
        history[3] = next_state_not_processed
        next_state = preprocess(history)
        prepross_history = next_state


def act(state,model):

    next_action = model.predict(state)

    return np.argmax(next_action[0])  # returns action

def main():
    test('first_model.h5',1000)
