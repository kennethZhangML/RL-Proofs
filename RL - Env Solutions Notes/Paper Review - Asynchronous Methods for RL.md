
**Problems to Solve**
- DNNs are highly unstable because samples are highly correlated
- Experience replay can be memory greedy when it comes to decorrelating samples

**Proposed Methods**
1. Asynchronously execute multiple agents on instances of the environment 
	1. fix problems and improve training efficiency 
	2. Decorrelates data collected from different states of the agents
	3. avoid using experience replay 
2. enables training on standard multi-core CPU -> requiring less resources 
3. Reduce resource usage and requirements 
4. observe actor-learners running in parallel 
	1. compute gradient of the network and update target network every M steps 
	2. Give each thread a different exploration policy 

![[Pasted image 20230618142131.png]]

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, output_size)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value

# Define the worker process
def worker(worker_id, global_model, optimizer, num_episodes):
    env = gym.make('CartPole-v0')  # Replace with your desired environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    local_model = ActorCritic(state_size, action_size)
    local_model.load_state_dict(global_model.state_dict())
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Synchronize local model with global model
            local_model.load_state_dict(global_model.state_dict())
            
            # Choose an action based on the local model's policy
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            policy, value = local_model(state_tensor)
            action = torch.multinomial(policy, num_samples=1).item()
            
            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if not done:
                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
                _, next_value = local_model(next_state_tensor)
            else:
                next_value = torch.tensor([0])
            
            # Compute the TD error and update the global model
            target = reward + 0.99 * next_value
            td_error = target - value
            loss = td_error.pow(2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        print(f"Worker {worker_id}, Episode {episode + 1}, Reward: {episode_reward}")
    
    env.close()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')  # Replace with your desired environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    global_model = ActorCritic(state_size, action_size)
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)
    
    num_workers = mp.cpu_count()  # Use the number of available CPU cores
    num_episodes = 100  # Number of episodes to run
    
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(worker_id, global_model, optimizer, num_episodes))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

```