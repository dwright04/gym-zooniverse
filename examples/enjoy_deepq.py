import gym
import argparse
import gym_zooniverse

from baselines import deepq

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='SegmentationTestEnv-v0')
    parser.add_argument('--model', help='saved deepq model', default='SegmentationTestEnv-v0.pkl')
    
    args = parser.parse_args()
    
    env = gym.make(args.env)
    act = deepq.load(args.model)
    env.reset()
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
