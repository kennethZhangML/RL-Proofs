
**Two Types of Value-based Functions**
1. State value function -> outputs expected return if agent starts at state, and acts according to the policy forever after 
2. Action value function -> outputs expected return if agent starts at state, and takes a given action at that state, then acts according to the policy
3. Value based methods -> optimal value function discovery => optimal policy

**Value Based Methods -> learn value, map state to expected value of being at that state**
1. value of state is expected discounted return -> start at state, then acts according to our policy
2. Train value function to output value of a state or a state action pair 
3. Create greedy policy -> always lead to greatest reward 

**Bellman Equation: Simplify our Value Estimation**
1. Simplifies state value or state-action value calculation
2. Recursive equation: any state = immediate reward Rt+1 plus discounted value of state that follows (gamma * V(st+1))
3. sum of immediate reward + discounted value of state that follows 

**Monte Carlo (end of Episode) -> calculates return and uses it as target for updating** 
1. Requires complete episode of interaction before updating value function
2. At the end, we have list of states, actions, rewards, and new state 
	1. uses this to update value function 
![[Pasted image 20230605224018.png]]

**Temporal Difference Learning (stepwise learning)**
1. waits for one interaction to form target and update value function
2. TD updates value function at each step 
3. estimate returns by adding reward of next state and discounted value of next state 
	1. bootstrapping -> based on existing estimate and not complete sample 

**![[Pasted image 20230605224005.png]]**


**Q Learning Algorithm**
1. Off policy value based method using TD approach to train action-value function
2. Q function: action value function determining value of being in a state and taking a specific action at that state 
3. State-action pair is cumulative reward our agent gets if it starts at this state then acts accordingly to its policy
4. Q-table: each cell corresponds to state-action pair value 
	1. cheat sheet for Q-function 
	2. initially zero initialized -> Q function searches inside Q-table to output value for a given state-action pair in the cells 
5. Train Q-function (given Q-table with all state-action pairs) and searches for value for state-action pair -> training completed implies optimal Q-table is found :
6. therefore we know the best action to take in each state
7. exploration trade-off allows for better approximation to optimal policy 

**Step-By-Step explanation of Q-learning Algorithm**
1. Initialize Q-table -> 0 initialized 
2. Choose action w/epsilon-greedy strategy 
	1. 1 - E = greedy action (probability of 1 - E -> select action with highest s-a value)
	3. E = random action (probability of E -> select random action) \
	4. Reduce Epsilon value 
3. Perform action given state, get next state reward and next state
4. Update Q(St, At) -> Q-function for state-action pair (TD approach)
	1. Immediate reward + discounted value of next state 
5. Select new state and action based on new state using epsilon-greedy 

```python 
import numpy as np 

def q_learning(env, num_episodes, learning_Rate, discount_factor, epsilon):
	NUM_STATES = env.observation_space.n 
	NUM_ACTIONS = env.action_space.n
	Q = np.zeros((NUM_STATES, NUM_ACTIONS))

	for episode in range(num_episodes):
		state = env.reset()
		done = False 

		while not done:
			if np.random.uniform(0, 1) < epsilon:
				action = env.action_space.sample()
			else:
				action = np.argmax(Q[state, :])
			next_state, reward, done, __ = env.step(action)
			Q[state, action] = (1 - learning_Rate) * Q[state, action] +
							(reward + discount_factor, * np.max(Q[next_state, :]))
			state = next_state 
		epsilon *= 0.99
	return Q 
```
