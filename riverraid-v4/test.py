import gym

import agent
import params


def test(env, net, device):
    env.metadata['render_fps'] = params.FPS
    env_name = env.unwrapped.__str__().replace('<', '').replace('>', '').split('/')[1]
    video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env.unwrapped, f'{env_name}.mp4')
    ag = agent.Agent(env, mode=agent.AgentMode.TEST, device=device, net=net)
    while not ag.test_step():
        video_recorder.capture_frame()
    video_recorder.close()
