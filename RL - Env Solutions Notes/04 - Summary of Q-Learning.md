-------------------------------------------------------------------------------
In this section we will build mathematical intuition for a summary of Q-learning. To mathematically explain Q-learning, we must define several key concepts:

1. Markov Decision Process: models decision making problems as a Tuple 
	1. (S, A, P, R):
		1. S - states of the environment
		2. A - actions the agent can take
		3. P - state transition probability matrix which defines probability of transitioning from state $s$ to state $s'$ when taking action $a$
		4. R - reward function, maps state-action pairs to immediate rewards

2. Value Function: estimates expected cumulative reward an agent can achieve from a particular state or state-action pair (state-value function or action value function)
	1. State Value function: expected cumulative reward in state $s$ and following a given policy
	2. Action Value function: expected cumulative reward in state $s$, taking action $a$, and following a given policy (e.g., neural network)

3. Q-value Iteration: estimate optimal action value function 
	1. Update the Q-value in Q-table via Bellman Equation
		1. Bellman Equation: optimal Q-value for a state-action pair is equal to:
			1. immediate reward + max Q-value of next state reached by taking corres. $a$
				$Q^*(s, a) = R(s, a) + \gamma * max(Q^*(s', a'))$ 
				- where, $\gamma$ is the discount factor 

4. Exploration-Exploitation Trade-off
	1. agent needs to explore different actions to gather info about the environment while also exploiting it's current knowledge to maximize rewards
	2. employs exploration strategy, s.t. epsilon greedy or softmax, to balance E&E

5. Q-learning Algorithm
	1. uses concept of temporal difference learning to iteratively update Q-value based on observed rewards and state transitions
	2. **At each step**, the agent observes the current state, takes an action based on its exploration strategy, receives a reward, and transitions to the next state
	3. Q-value update equation for Q-learning is:
			$Q^*(s, a) \leftarrow Q(s, a) + \alpha * [R + \gamma * max(Q(s', a')) - Q(s, a)]$
			where alpha is the learning rate determining the weight given new info compared to existing Q-values in the Q-table 

6. Convergence and Optimality
	1. Q-values in Q-learning converge to optimal action values, $Q^*(s, a)$, which represent the maximum expected cumulative reward achievable in each state-action pair
	2. Once optimal Q-values are determined, optimal policy is derived by getting the actions with the highest Q-value in each state
-------------------------------------------------------------------------------


	 