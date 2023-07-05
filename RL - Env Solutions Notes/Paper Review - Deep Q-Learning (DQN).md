
"Playing Atari with Deep Reinforcement Learning" - Mnih et Al.
![[Pasted image 20230607225528.png]]
**Problems**
- high dim sensory inputs
- noisy delayed algorithm divergences 
- highly correlated states, non stationary distribution of data 

**Proposed Methods**
- **NN function approximator -**> eliminate non-scalability of Q-table 
	- weight (theta) as action-value function approximator 
	- SGD update weights optimization 
- **Experience Replay**
	- store experiences as state transitions pooled over episodes
	- smoothing learning function -> avoid divergence and oscillation of action values
	- at each update -> sample minibatch of state transitions from replay memory
		- eliminates correlations between states -> reduce variance of updates
	- perform minibatch gradient descent on mean squared error between:
		- state action value
		- Temporal Dependency Target
	- get rid of unwanted feedback loops or poor local minima 
	- reduce affects of biased policy by dominating samples 
- **Reward Clipping** -> limit scale of error derivatives 
	- improves learning rate scalability over multiple games
- **Frame Skipping** -> select every nth frame instead of every frame -> play more games without increasing too much runtime 


