from mcts import mcts
from utils import env_loader
from multiprocessing import freeze_support
import datetime
import copy


if __name__ == '__main__':
    freeze_support()

    env = env_loader.load_rooms_env()
    agent = mcts.MCTS(env.action_space)

    TOTAL_EPISODES = 5
    scores = []

    for episode in range(TOTAL_EPISODES):
        state = env.reset()
        discounted_return = 0
        steps = 0
        done = False
        print('Episode {} of {}'.format(episode + 1, TOTAL_EPISODES))
        start = datetime.datetime.utcnow()

        while not done:
            steps += 1
            action = agent.policy(copy.deepcopy(env))
            next_state, reward, done, _ = env.step(action)
            discounted_return += reward * (0.99 ** steps)
            if done:
                end = datetime.datetime.utcnow()
                print('done after {} steps, duration: {}'.format(steps, end-start))
                scores.append(discounted_return)
                break

    for score in scores:
        print('Discounted Return from episode {} of {}: {}'.format(scores.index(score) + 1, TOTAL_EPISODES, score))

    print('Average discounted return: {}'.format(sum(scores) / len(scores)))
