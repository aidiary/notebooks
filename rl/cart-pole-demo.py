import gym
import numpy as np


def gen_random_policy():
    return (np.random.uniform(-1, 1, size=4), np.random.uniform(-1, 1))


def policy_to_action(env, policy, obs):
    if np.dot(policy[0], obs) + policy[1] > 0:
        return 1
    else:
        return 0


def run_episode(env, policy, t_max=1000, render=False):
    obs = env.reset()
    total_reward = 0
    for i in range(t_max):
        if render:
            env.render()
        selected_action = policy_to_action(env, policy, obs)
        obs, reward, done, _ = env.step(selected_action)
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # Generate a pool or random policies
    n_policy = 100
    policy_list = [gen_random_policy() for _ in range(n_policy)]

    # Evaluate the score of each policy.
    scores_list = [run_episode(env, p) for p in policy_list]

    # Select the best plicy.
    print('Best policy score = %f' % max(scores_list))

    best_policy = policy_list[np.argmax(scores_list)]
    print('Running with best policy:\n')
    run_episode(env, best_policy, t_max=5000, render=True)
