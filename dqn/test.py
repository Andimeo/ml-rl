import gym

from dqn import agent


def test(env, params, net, device):
    env.metadata['render_fps'] = params.FPS
    env_name = env.metadata['name']
    video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env.unwrapped, f'{env_name}.mp4')
    ag = agent.Agent(env, params, mode=agent.AgentMode.TEST, device=device, net=net)
    while not ag.test_step():
        video_recorder.capture_frame()
    video_recorder.close()
