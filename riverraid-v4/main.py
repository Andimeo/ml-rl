import torch

import agent
import dqn
import test
import train
import wrappers


def main(mode, device):
    if mode is agent.AgentMode.TRAIN:
        env = wrappers.make_env(
            'ALE/Riverraid-v5', episode_life=True, clip_reward=True)
        train.train(env, device)
    else:
        env = wrappers.make_env('ALE/Riverraid-v5', frame_stack=True, render_mode='human')
        net = dqn.DQN(env.observation_space.shape, env.action_space.n)
        net.load_state_dict(torch.load(
            '/Users/andimeo/Downloads/Riverraid-3446.00.dat', map_location=torch.device('cpu')))
        test.test(env, net, device)
    env.close()


if __name__ == '__main__':
    main(mode=agent.AgentMode.TRAIN, device='cuda')
