
**Proposal**
- NN architecture to aid with better policy learning without changing underlying RL algorithm
- Modifies original DQN by introducing two new streams 

**Methodology**
- Duelling Architecture -> DQN with two streams of FC layers after conv ops 
- Stream 1: state value estimator 
- Stream 2: advantage calculation for each action

![[Pasted image 20230609185935.png]]

**Aggregation**: aggregating module 
![[Pasted image 20230609190008.png]]
- uses the sum of the state value estimation and the action advantage calculation
- updates the Q-function of the DQN 

**Unidentifiability: Q-value cannot recover value and action uniquely** 
add a constant to thee Value function and subtract the same constant from the action-selection process -> this constant cancels out resulting in the same Q-value 
- force the advantage function estimator to have zero advantage at the chosen action 

**Improvement: force advantage function estimator to have zero advantage** 
- the value estimator function provides an estimate of the value function while the other stream provides an estimate of the advantage function 

**Result: Separation of state and advantage estimation**
By separating the state and advantage estimation, the duelling architecture can learn which states are valuable without having to learn the effect of each action for each state 

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random

# Step 1: Set up the environment (assumed to be a discrete action space environment)

# Step 2: Define the neural network architecture

class DuellingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuellingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2_adv = nn.Linear(64, 64)
        self.fc2_val = nn.Linear(64, 64)
        self.fc3_adv = nn.Linear(64, output_size)
        self.fc3_val = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        adv = torch.relu(self.fc2_adv(x))
        val = torch.relu(self.fc2_val(x))
        adv = self.fc3_adv(adv)
        val = self.fc3_val(val).expand(x.size(0), self.fc3_adv.out_features)
        return val + adv - adv.mean(dim=1, keepdim=True)

# Step 3: Implement the replay buffer

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Step 4: Define the loss function

def compute_loss(batch, model, target_model, gamma):
    states, actions, rewards, next_states, dones = batch

    # Compute Q-values
    Q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q-values using the target network
    next_Q_values = target_model(next_states).max(1)[0].detach()
    target_Q_values = rewards + (gamma * next_Q_values * (1 - dones))

    # Compute loss
    loss = nn.MSELoss()(Q_values, target_Q_values)
    return loss

# Step 5: Implement the training loop

def train(env, model, target_model, optimizer, replay_buffer, batch_size, gamma, epsilon, epsilon_decay, target_update):
    total_steps = 0
    episode_rewards = []
    epsilon = epsilon
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Select an action based on epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store transition in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) >= batch_size:
                # Sample a batch of transitions from the replay buffer
                batch = replay_buffer.sample(batch_size)

                # Calculate the loss and perform a gradient descent step
                optimizer.zero_grad()
                loss = compute_loss(batch, model, target_model, gamma)
                loss.backward()
                optimizer.step()

                # Update target network weights
                if total_steps % target_update == 0:
                    target_model.load_state_dict(model.state_dict())

            total_steps += 1

        episode_rewards.append(total_reward)
        epsilon *= epsilon_decay

    return episode_rewards

# Step 6: Test the trained network

def test(env, model):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            q_values = model(torch.tensor(state, dtype=torch.float32))
            action = q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

# Set up the environment

# Define hyperparameters
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
target_update = 1000
num_episodes = 1000
capacity = 10000
learning_rate = 0.001

#

```
