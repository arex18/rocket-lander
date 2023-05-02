import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.rng = np.random.default_rng()

    def store_transition(self, state, action, reward, state_next, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_next
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done # stored as 1-done let's us use reward + future_reward*done <-- (ie. will future reward will be 0 in terminal state)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = self.rng.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

class Critic(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg', model_name='_ddpg.chkpt'):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        current_dir = os.getcwd()
        self.checkpoint_file = os.path.join(current_dir, chkpt_dir, name + model_name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3=0.003
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))  # around 22:00
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save({'state':self.state_dict(),
                'optimizer':self.optimizer.state_dict},
                 self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        checkpoint = T.load(self.checkpoint_file)
        self.load_state_dict(checkpoint['state'])
        self.optimizer.load_state_dict(checkpoint['optimizer']())


class Actor(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg', model_name='_ddpg.chkpt'):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        current_dir = os.getcwd()
        self.checkpoint_file = os.path.join(current_dir, chkpt_dir, name + model_name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3=0.003
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save({'state':self.state_dict(),
                'optimizer':self.optimizer.state_dict},
                 self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        checkpoint = T.load(self.checkpoint_file)
        self.load_state_dict(checkpoint['state'])
        self.optimizer.load_state_dict(checkpoint['optimizer']())


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=3, max_size=int(1e6),
                layer1_size=400, layer2_size=300, batch_size=64, chkpt_dir='tmp/ddpg', model_name='_ddpg.chkpt', test_phase=False):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.test_phase = test_phase
        self.actor = Actor(alpha, input_dims, layer1_size, layer2_size,
                           n_actions=n_actions, name='Actor', chkpt_dir=chkpt_dir, model_name=model_name)

        self.target_actor = Actor(alpha, input_dims, layer1_size, layer2_size,
                           n_actions=n_actions, name='TargetActor', chkpt_dir=chkpt_dir, model_name=model_name)

        self.critic = Critic(beta, input_dims, layer1_size, layer2_size,
                           n_actions=n_actions, name='Critic', chkpt_dir=chkpt_dir, model_name=model_name)

        self.target_critic = Critic(beta, input_dims, layer1_size, layer2_size,
                           n_actions=n_actions, name='TargetCritic', chkpt_dir=chkpt_dir, model_name=model_name)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):   # 34:00
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        if self.test_phase:
            mu_prime = mu.to(self.actor.device)
        else:
            mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            self.actor.train()

        return mu_prime.cpu().detach().numpy()

    def act(self, env, observation):   # same function, renamed for Evaluation Framework
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        if self.test_phase:
            mu_prime = mu.to(self.actor.device)
        else:
            mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            self.actor.train()

        return mu_prime.cpu().detach().numpy()

    def draw(self, env):  # Just here to satisify evaluation framework
        return None

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    '''
    DDPG Algorithm
        notation:
        theta^Q: critic parameters
        theta^mu: actor parameters
        theta^Q': critic target params
        theta^mu': actor target params
        N: Noise process, R: Replay buffer

        - Initalize critic Q(s,a|theta^Q) and actor mu(s|theta^mu) networks with weights theta^Q and theta^mu
        - initalize target critic and target actor networks with weights theta^Q' <-- theta^Q and theta^mu' <-- theta^mu
        - Initalize replay buffer R
        for episodes:
            Initalize a random process (OU Noise) N for action exploration
            Get initial state s_1
            for steps:
                Select action a_t = mu(s_t|theta^mu) + N_t
                Execute a_t get reward r_t and new state s_t+1
                Store transition in R
                Sample mini-batch of n transitions from R
                Set y_i = r_i + gamma*Q'(s_i+1,mu'(s_i+1|theta^mu')|theta^Q')
                Update Critic by minimizing Loss
                Update Actor policy using sampled policy gradient
                Update target networks using polyak value: rho
                    theta^Q' <-- rho*theta^Q + (1-rho)*theta^Q'
                    theta^mu' <-- rho*theta^mu + (1-rho)*theta^mu'

    '''

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_target = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_target[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target.unsqueeze(1), critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        # # dict comprehension
        # target_critic_update = {k:tau*critic_state_dict[k].clone() + (1-tau)*v.clone() for (k,v) in target_critic_dict.items()}
        # target_actor_update = {k:tau*actor_state_dict[k].clone() + (1-tau)*v.clone() for (k,v) in target_actor_dict.items()}
        # self.target_critic.load_state_dict(target_critic_update)
        # self.target_actor.load_state_dict(target_actor_update)

        # update target networks
        for name in target_critic_dict.keys():
            target_critic_dict[name] = tau*critic_state_dict[name].clone() + \
                                        (1-tau)*target_critic_dict[name].clone()

        for name in target_actor_dict.keys():
            target_actor_dict[name] = tau*actor_state_dict[name].clone() + \
                                        (1-tau)*target_actor_dict[name].clone()

        self.target_critic.load_state_dict(target_critic_dict)
        self.target_actor.load_state_dict(target_actor_dict)

# # Rework this:
#         for name in target_critic_dict.keys():
#             target_critic_dict[name] = tau*critic_state_dict[name].clone() + \
#                                         (1-tau)*target_critic_dict[name].clone()

#         for name in target_actor_dict.keys():
#             target_actor_dict[name] = tau*target_actor_dict[name].clone() + \
#                                         (1-tau)*actor_state_dict[name].clone()

#         self.target_critic.load_state_dict(target_critic_dict)
#         self.target_actor.load_state_dict(target_actor_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()