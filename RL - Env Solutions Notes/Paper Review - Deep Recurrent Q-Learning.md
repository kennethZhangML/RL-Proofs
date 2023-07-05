
Deep Recurrent Q-Learning for Partially Observable MDPs

**Original DQN limitations**
1. Assumes that game state is fully observable 
2. Limited memory to 4 frames 
3. Requires memorizing more that four frames is difficult 
4. real-world tasks are usually only partially observable and state info is incomplete and noisy

**Partial Observability**
1.  States, actions, transitions, rewards -> represents one POMDP observation
2. estimating Q-value from observation can be bad given DQN, given limitations 

**Proposed Network Architecture**
- LSTM Architecture + DQN 
- input images conv 3x -> go via LSTM layer -> final linear layer 
	- outputs Q-value for each action
- LSTM layer integrates info via time -> learning via transition trajectories 
	- help with evaluations of Q-values under partial observations 

![[Pasted image 20230608220835.png]]

- DRQN -> given single frame, can make decisions satisfactorily 
- outperforms DQN especially when state info is incomplete 

**MDP to POMDP Generalization**
- DRQN degrades better than DQN 
	- Pong MDP vs Flickering Pong 
	- Recurrent controllers have certain degree of robustness against missing info 

