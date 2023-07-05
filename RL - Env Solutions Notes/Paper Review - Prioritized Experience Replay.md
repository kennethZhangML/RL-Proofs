
**Main Concepts**
- Sample memories with higher significance from replay buffer instead of sampling uniformly

**Proposed Methods**
- RL agents learn better from particular transitions
- Agent Temporal Dependency error can be used to measure priority of each transition, since it shows how surprising or unexpected the transition is 
- **Greedy Prioritization**
	- select memories with highest TD error for every update
	- binary heap data structure for priority queue 
	- Low TD error transitions may not be replayed for long time 
	- sensitive to noise spikes -> fail easily when there are approximation errors 
	- focuses only on subset of the experience -> high errors 
- **Stochastic Prioritization**
	- overcomes issues from greedy prioritization -> sampling method 
	- probability of sampling transition is: ![[Pasted image 20230615195142.png]]
	- alpha determines the quantity of prioritization being used (a = 0 being uniform)
	- Proportional Prioritization -> sum of absolute sigma and epsilon value 
	- Rank-based prioritization -> rank when replay memory is sorted according to sigma

**Implementation Guide**
- rank-based prioritization -> approximate cumulative density function with piecewise linear function with k-segments of equal probability
- proportional prioritization -> sum-tree data structure of implementation
	- sum of the nodes in a binary tree 

**Annealing Bias**
- estimating expected value with stochastic updates 
	- same distribution as its expectation 
- prioritized replay introduces bias -> corrected using importance-sampling weights
	- corrects non-uniform probability -> integrated by normalizing the weights 
	- anneal linear units -> increasing beta coefficient 

![[Pasted image 20230617100957.png]]

```python
import numpy as np
from collections import deque

class PrioritizedExperienceReplay:
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, epsilon=0.01):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.priorities_sum = 0.0
        self.max_priority = 1.0

    def add_experience(self, experience, priority):
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.priorities_sum += priority
        if priority > self.max_priority:
            self.max_priority = priority

    def sample_batch(self, batch_size):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha / self.priorities_sum
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)

        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            priority = (error + self.epsilon) ** self.alpha
            self.priorities_sum -= self.priorities[i]
            self.priorities[i] = priority
            self.priorities_sum += priority
            self.max_priority = max(self.max_priority, priority)
            
    def get_max_priority(self):
        return self.max_priority

    def __len__(self):
        return len(self.buffer)

```
