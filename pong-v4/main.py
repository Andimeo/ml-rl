import agent
import dqn
import test
import train
import wrappers

import torch

def main(mode, device):
    if mode is agent.AgentMode.TRAIN:
        env = wrappers.make_env('PongNoFrameskip-v4')
        train.train(env, device)
    else:
        env = wrappers.make_env('PongNoFrameskip-v4', render_mode='human')
        net = dqn.DQN(env.observation_space.shape, env.action_space.n)
        net.load_state_dict(torch.load(
            'models/pong-19.8.dat', map_location=torch.device('cpu')))
        test.test(env, net, device)
    env.close()


if __name__ == '__main__':
    main(mode=agent.AgentMode.TRAIN, device='cuda')
