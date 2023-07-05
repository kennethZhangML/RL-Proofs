	
**Q-learning Issues:** 
1. Not scalable -> tabular method is not scalable if action and states spaces are not small enough to be represented efficiently by arrays and tables
2. Q-table is not scalable -> use neural network to approximate with weights
	1. Extract high level features of the high-dimensional sensory input 
	2. Updated using Q-learning and SGD to update weights 
3. Mnih et Al, proposed experience replay mechanism -> agent stores experience as state transitions pooled over many episodes into a replay memory 
	1. randomly samples minibatches of state transitions from memory during update
	2. minibatch gradient descent on mean squared error between state-action value and Temporal Dependent target
	3. Parameterized Q-function approximated by neural network is resulting 

**Network Architecture:**
1. Input: Stack of 4 frames passed via network as state
	1. output is vector of Q-values for each possible action for that state 
2. Resize and reduce complexity of state -> reduce to 84x84 (grayscale)
	1. Reduce from RGB to 1 colour scale 
	2. Stack of 4 frames -> exploits spatial relationships in images 
	3. Exploit temporal properties 
3. Output Q-value for each possible action of that state 

**Deep Q-Learning Algorithm Training** 
Loss function compares Q-value prediction and Q-target using grad descent 
Disadvantages: instability from combining non-linear Q-value function and bootstrapping 
Two phases: sampling and training 
1. Sampling -> store observation experience tuples in replay memory
2. Training -> select minibatch random sample of tuples and update using grad descent 

![[Pasted image 20230607223947.png]]

**How to Stabilize Deep Q-Learning Algorithm**
1. Experience replay -> use of past experiences
	1. reuse experiences during training -> learn from same experiences multiple times 
	2. replay buffer that saves tuples of experiences then sample batches of the tuples
	3. stochastic sampling -> remove correlation and avoid divergence of action values
	4. initialize replay memory buffer D with capacity N 
		1. N is number of experiences in replay memory
		2. D is buffer size 
	5. store experiences in memory and sample batches of experiences to feed DQN 
2. Fixed Q-Target -> difference between target and current Q-value 
	1. no idea of what the TD target is -> estimation is necessary 
	2. correlation between TD target and changing parameters -> q-values and targets shift
	3. use separate network with fixed parameters for estimating TD target 
	4. copy parameters from DQN to every step to update target network 
3. Double DQN: handle overestimation of Q-values 
	1. Use DQN to choose best next action to take 
	2. Use target network to calculate the target Q-value for taking that action at next state