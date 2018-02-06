# Utilisation de OpenAI Gym pour avoir l'agent
import sys
sys.path=['',
 '/opt/anaconda2/lib/python27.zip',
 '/opt/anaconda2/lib/python2.7',
 '/opt/anaconda2/lib/python2.7/plat-linux2',
 '/opt/anaconda2/lib/python2.7/lib-tk',
 '/opt/anaconda2/lib/python2.7/lib-old',
 '/opt/anaconda2/lib/python2.7/lib-dynload',
 '/opt/anaconda2/lib/python2.7/site-packages',
 '/opt/anaconda2/lib/python2.7/site-packages/Sphinx-1.5.1-py2.7.egg',
 '/opt/anaconda2/lib/python2.7/site-packages/biopython-1.69-py2.7-linux-x86_64.egg',
 '/opt/anaconda2/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg',
 '/opt/anaconda2/lib/python2.7/site-packages/IPython/extensions',
 '/home/tp-home001/gcolli2/.ipython']
 
import gym
import time
from DQL import DQLAgent
from preprocess import preprocess
from keras.models import load_model
import pandas as pd
import numpy as np

env = gym.make('Breakout-v0')
state_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQLAgent(state_size, action_size)

######## A changer
file = 'third_model'

if __name__ == '__main__':
    nb_episode = 1000000
    nb_step = 1000
    batch_size = 32
    history = [0, 0, 0, 0]
    save_models_each = 10000
    restart = True
    no_training_steps = 100000
    allow_replay = False
    load_model_from_file = 'second_model'
    average_score = 0
    list_average_score = []

    if not restart:
        # On charge le dernier modèle
        agent.continue_model(load_model_from_file)
        # Et les résultats
        save_average_score = pd.read_csv('output/' + file + '.csv')
    else:
        save_average_score = pd.DataFrame({'Episode': [0],
                                           'Average_score': [0]})


    for episode in range(nb_episode):
        print('Début du ' + str(episode) + ' épisode')
        state = env.reset()
        history[0] = state
        history[1] = state
        history[2] = state
        history[3] = state
        compteur = 0
        score = 0
        prepross_history = preprocess(history)
        for step in range(nb_step):
            compteur += 1
            # Uncomment to unable rendering
            # env.render()
            # slepp is needed to correct an issue with pyglet/ mac os only ?
            # time.sleep(0.1)

            # on cherche la meilleure action

            action = agent.act(prepross_history)

            save_state = prepross_history

            # On la fait
            next_state_not_processed, reward, done, info = env.step(action)
            score += reward

            history[0:3] = history[1:4]
            history[3] = next_state_not_processed

            next_state = preprocess(history)
            prepross_history = next_state
            # On la sauvegarde
            agent.add_to_memory(state=save_state, action=action, reward=reward, next_state=next_state, terminal=done)

            if done:
                message = 'Agent has reached the end in {} steps with score {}'.format(compteur, score)
                print(message)
                break
            state = next_state

            if allow_replay:
                agent.replay(batch_size)
        average_score += score
        if episode % save_models_each == 0:
            agent.save('output/backup.h5')
            list_average_score.append((nb_episode, average_score / 1000))
            average_score = 0
            print('Le score moyen sur les 1000 derniers épisode était {}'.format(average_score))
        if no_training_steps > 0:
            no_training_steps -= compteur
        else:
            allow_replay = True
    save_average_score[0] = np.array(list_average_score)[:, 0]
    save_average_score[1] = np.array(list_average_score)[:, 1]
    save_average_score.to_csv('output/' + file + '.csv')
    agent.save('output/' + file + '.h5')
