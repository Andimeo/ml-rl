import cv2
import gym
import numpy as np


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        shape = (2, ) + env.observation_space.shape
        self._obs_buffer = np.zeros(shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = info = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process2(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, 'Unknown resolution.'

        img = img[:, :, 0] * 0.299 + img[:, :, 1] * \
            0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

    @staticmethod
    def process2(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        size = (84, 84)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, -1)
        return frame


class BufferWrapper(gym.Wrapper):
    def __init__(self, env, n_steps):
        super().__init__(env)
        self.n_steps = n_steps
        shp = env.observation_space.shape
        shape = (shp[0] * n_steps, ) + shp[1:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=env.observation_space.dtype
        )
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for i in range(self.n_steps):
            self.buffer[i] = np.squeeze(ob, axis=0)
        return np.array(self.buffer)
        # return np.array(self.frames)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = np.squeeze(ob, axis=0)
        return np.array(self.buffer), reward, done, info


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(
            old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def make_env(env_name, render_mode=None):
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = MaxAndSkipEnv(env)
    env = BufferWrapper(env, 4)
    # env = ScaledFloatFrame(env)
    return env
