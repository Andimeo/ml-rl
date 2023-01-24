import gym
import numpy as np
import torch

import agent


def test(env, net, device):
    video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, 'pong.mp4')
    ag = agent.Agent(env, mode=agent.AgentMode.TEST,
                     device=device, net=net, video_recorder=video_recorder)

    while not ag.test_step():
        pass
    video_recorder.close()