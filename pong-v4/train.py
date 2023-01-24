import time

import numpy as np
import torch

import agent
import params


def train(env, device):
    ag = agent.Agent(env, mode=agent.AgentMode.TRAIN, device=device)
    epsilon = params.EPSILON_START

    total_rewards = []
    frame_idx = 0
    best_mean_reward = None
    ts = time.time()
    while True:
        reward = ag.play_step(epsilon)
        epsilon = max(epsilon - params.EPSILON_STEP, params.EPSILON_FINAL)

        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-10:])
            speed = frame_idx / (time.time() - ts)
            print('%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s' %
                  (frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(ag.net.state_dict(),  "Pong-best.dat")
                if best_mean_reward is not None:
                    print(f"Best mean reward updated {best_mean_reward:.3f} -> {mean_reward:.3f}, model saved")
                best_mean_reward = mean_reward
            if mean_reward > params.MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if frame_idx > params.REPLAY_START_SIZE:
            ag.train()
            if frame_idx % params.SYNC_TARGET_FRAMES == 0:
                ag.sync_nets()
        frame_idx += 1
    return ag.net
