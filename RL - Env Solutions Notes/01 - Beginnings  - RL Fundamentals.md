1. **The RL Loop - Outputs sequence of data (OpenAI Gym)**
	1. State, action, reward, and next state 
	2. Maximize cumulative reward (expected reward)
	
2. **Reward Hypothesis -> Maximization of Expected Return** 
	1. To have the best behaviour we aim to learn to take actions that maximize the expected cumulative reward (via the Reward Hypothesis)
	
3. **Key Property: Markov Property** -> only the current state is needed to decide what action to take next **(not all history of states and actions is required)**

4. **Observation and State Spaces -> info our agent gets from the environment** 
	1. State: complete description of the state of the world
	2. Observation: partial description of the state 
	Example: In Mario, we only see part of the level close to the player so we receive an observation
	 
1. **Action Space: all possible actions in an environment** 
	1. Discrete (finite number of actions)
	2. Continuous (infinite number of actions)

2. **Rewards and Discounting**
	1. Fundamental, only feedback for the agent -> good or bad actions
	2. Cannot simply be added -> probabilities need to be considered based on factors contributed by the environment the agent is in 
		1. Introduce **discount rate (gamma)** -> future reward is less and less likely to happen
		2. our agent cares more about the long term reward -> larger gamma 
		3. our agent cares more about the short term reward -> smaller gamma 

3. **Types of Tasks**
4. Episodic Tasks -> start and end point
	1. List of states, actions, rewards, and new states 
5. Continuing Tasks -> no terminal state -> choose best actions and simultaneously interact with the environment -> agent keeps running until we decide to stop it 

6. **Exploration and Exploitation Trade Off**
	1. Exploration: find more info about the environment 
	2. Exploitation: Known info to maximize the reward 
7. **Common trap: hit or miss chance** -> could result in fatal risk taking 
	1. Must find exploration and exploitation balance

8. **Policy** - The Function we wish to learn 
	1. Find optimal policy to maximize expected return when agent acts according to it
	2. We find the pi* through training loops 
9. **Direct -> policy-based methods** "Map State to best corres action"
	1. Learn Policy function directly -> mapping from each state to best corres. action
	2. Defined over probability distribution over set of possible actions at that state 
		1. Deterministic: will always return the same action
		2. Stochastic: outputs a probability distribution over actions 
10. **Indirect -> Value-Based methods** "Act according to our policy - go to state with highest value"
	1. Learn function that maps state to expected value of being at that state 
	2. Value of state: expected discounted return the agent can get if it starts in that state and then acts according to that policy 


