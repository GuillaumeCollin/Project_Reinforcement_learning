#Utilisation de OpenAI Gym pour avoir l'agent

import gym
import time
from DQL import DQLAgent
from preprocess import preprocess
from keras.models import load_model

env = gym.make('Breakout-v0')
state_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQLAgent(state_size,action_size)
env.reset()



if __name__ == '__main__':
    nb_episode = 10000
    nb_step = 1000
    batch_size = 32
    history = [0,0,0,0]


    for episode in range(nb_episode):
        print('Début du ' + str(episode) + ' épisode')
        state = env.reset()
        history[0] = state
        history[1] = state
        history[2] = state
        history[3] = state
        compteur = 0
        prepross_history = preprocess(history)
        for step in range(nb_step):
            compteur += 1
            #Uncomment to unable rendering
            #env.render()
            # slepp is needed to correct an issue with pyglet/ mac os only ?
            #time.sleep(0.1)

            #on cherche la meilleure action

            action = agent.act(prepross_history)

            save_state = prepross_history

            #On la fait
            next_state_not_processed, reward, done, info = env.step(action)

            history[0:3] = history[1:4]
            history[3] = next_state_not_processed

            next_state=preprocess(history)
            prepross_history=next_state
            #On la sauvegarde
            agent.add_to_memory(state=save_state,action=action,reward=reward,next_state=next_state,terminal=done)

            if done:
                print('Agent has reached the end in ' + str(compteur) + ' steps')
                break
            state = next_state
        agent.replay(batch_size)
    agent.save('output/first_model.h5')