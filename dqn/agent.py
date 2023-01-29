import collections
import enum

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dqn


Experience = collections.namedtuple('Experience', field_names=[
                                    'state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        # indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


class AgentMode(enum.Enum):
    TRAIN = 'train'
    TEST = 'test'


class Agent:
    def __init__(self, env, params, mode=AgentMode.TRAIN, device='cpu', **kwargs) -> None:
        self.env = env
        self.batch_size = params.BATCH_SIZE
        self.mode = mode
        self.device = device

        if mode is AgentMode.TRAIN:
            self.exp_buffer = ExperienceBuffer(params.REPLAY_SIZE)
            self.net = dqn.DQN(env.observation_space.shape, env.action_space.n).to(device)
            self.tgt_net = dqn.DQN(env.observation_space.shape, env.action_space.n).to(device)
            self.sync_nets()
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=params.LEARNING_RATE)
            self.gamma = params.GAMMA
        else:
            assert 'net' in kwargs
            self.net = kwargs.get('net')
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def select_action(self):
        state_a = np.array([self.state], copy=False)
        state_v = torch.tensor(state_a).to(self.device)
        q_vals_v = self.net(state_v)
        # torch.max return value and index
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
        return action

    def test_step(self):
        action = self.select_action()
        state, _, is_done, _ = self.env.step(action)
        self.state = state
        return is_done

    def play_step(self, epsilon=0.0):
        done_reward = None

        rnd = np.random.random()
        if rnd < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.select_action()

        new_state, reward, is_done, _ = self.env.step(action)
        
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def calc_loss(self, batch):
        states, actions, rewards, dones, next_states = batch
        states_v = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)

        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.tgt_net(next_states_v).max(1)[0]  # value and index
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def train(self):
        batch = self.exp_buffer.sample(self.batch_size)
        loss_t = self.calc_loss(batch)

        self.optimizer.zero_grad()
        loss_t.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return float(loss_t.detach().cpu().numpy())

    def sync_nets(self):
        self.tgt_net.load_state_dict(self.net.state_dict())
