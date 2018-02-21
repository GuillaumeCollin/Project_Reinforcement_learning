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
 '/home/tp-home001/gcolli2/.ipython',
 '/home/tp-home001/gcolli2/.local/lib/python2.7/site-packages',]
 
import gym
import time
from DQL import DQLAgent
from preprocess import preprocess
import test
from keras.models import load_model
import pandas as pd
import numpy as np

######## A changer
nb_episode = 4000000
file = 'fifth_model'
restart = True

env = gym.make('Breakout-v0')
state_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQLAgent(state_size, action_size, nb_episode)


if __name__ == '__main__':
    batch_size = 32
    history = [0, 0, 0, 0]
    save_models_each = 300
    no_training_steps = 50000
    allow_replay = False
    load_model_from_file = file
    average_score = 0
    repeat_action = 4
    beginning_episodes = 0

    if not restart:
        # On charge le dernier modele
        agent.continue_model(load_model_from_file)
        # Et les resultats
        save_average_score = pd.read_csv('output/' + file + '.csv')
        beginning_episodes = int(save_average_score['Episode'][save_average_score.index[-1]])
        agent.update_epsilon(beginning_episodes)
    else:
        save_average_score = pd.DataFrame({'Average_score': [0],
                                           'Episode': [0]})

    for episode in range(beginning_episodes,nb_episode):
        print('Debut du ' + str(episode) + ' episode')
        state = env.reset()
        history = np.stack((state,state,state,state),axis=0)
        compteur = 0
        score = 0
        done = False
        prepross_history = preprocess(history)

        while not done:
            compteur += 1
            # Uncomment to unable rendering
            # env.render()
            # slepp is needed to correct an issue with pyglet/ mac os only ?
            # time.sleep(0.1)

            # on cherche la meilleure action

            action = agent.act(prepross_history)

            save_state = prepross_history

            # On la fait
            for action in range(repeat_action):
                next_state_not_processed, reward, done, info = env.step(action)
                score += reward

                history[0:3] = history[1:4]
                history[3] = next_state_not_processed

                next_state = preprocess(history)
            prepross_history = next_state
            # On la sauvegarde
            agent.add_to_memory(state=save_state, action=action, reward=reward, next_state=next_state, terminal=done)

            if allow_replay:
                agent.replay(batch_size)
        message = 'Agent has reached the end in {} steps with score {}'.format(compteur, score)
        print(message)
        average_score += score
        if episode % save_models_each == 0 and episode != beginning_episodes:
            agent.save('output/' + file + '.h5')
            save_average_score.loc[save_average_score.index[-1] + 1] = np.array([average_score / save_models_each, episode])
            save_average_score.to_csv('output/' + file + '.csv')
            print('Le score moyen sur les {} derniers episode etait {}'.format(save_models_each,
                                                                               average_score / save_models_each ))
            average_score = 0
        if no_training_steps > 0:
            no_training_steps -= compteur
        else:
            allow_replay = True

    print('Training is over')
    test(file, True)