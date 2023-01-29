import torch

import dqn
import games.breakout.params as params


def main(mode, device):
    if mode is dqn.AgentMode.TRAIN:
        env = dqn.make_env('ALE/Breakout-v5',
                           episode_life=params.EPISODE_LIFE,
                           frame_stack=params.FRAME_STACK,
                           clip_reward=params.CLIP_REWARD)
        dqn.train(env, params, device)
    else:
        env = dqn.make_env('ALE/Breakout-v5', frame_stack=params.FRAME_STACK, render_mode='human')
        net = dqn.DQN(env.observation_space.shape, env.action_space.n)
        net.load_state_dict(torch.load(
            '/Users/andimeo/Downloads/Breakout-v5-4.30.dat', map_location=torch.device('cpu')))
        dqn.test(env, params, net, device)
    env.close()


if __name__ == '__main__':
    main(mode=dqn.AgentMode.TRAIN, device='cuda')
