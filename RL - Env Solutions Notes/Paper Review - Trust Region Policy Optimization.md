
**Problems to Solve**
1. improvement process in policy gradient methods contain lots of inefficient update steps 
2. We should MINIMIZE surrogate objective function guaranteeing improvement in non-trivial step sizes -> via single and vine method 

**Trajectory Collection**
1. Interact with environment using initial policy 
	1. select actions based on current policy 
	2. observe results -> receives rewards from env

**Compute advantage estimates**
	1. using the trajectories -> calculate the advantages 
	2. advantages are the difference between the actual returns and predicted values of the states -> help quantify quality of chosen actions compared to expected return 

**Use Surrogate Loss for Policy Updates**
	1. difference between new policy and old policy
		1. based on collected trajectories and advantages
	2. encourages policy to take action with increased likelihood of actions with higher advantages -> designed to approximate natural policy gradient ascent 
	3. ensure policy updates are performed in direction of maximizing improvement 

**Steps for implementing Surrogate Loss**
1. Obtain logits via passing states through policy
	1. normalize via SoftMax -> represents action distribution for each state
2. Select probabilities from the new action distribution that correspond to actions taken during data collection -> create one-hot encoding of actions and applying element-wise multiplication
3. Compute ratio of new to old probabilities -> action to old probs 
4. Calculate surrogate loss
	1. first loss: ratios calculated previously multiplied by advantages
	2. 2nd loss: clipped trust region ratios multiplied by advantages
	3. final loss is minimum of 1st and 2nd surrogate loss over the batch average

**Calculate Policy Update Direction**
	1. policy update direction calculated via natural policy gradient
		1. considers geometry of policy space and constraints posed by trust region 
	2. constrains policy in trust region 

**Line Search**
	1. TRPO performs line search to determine step size for policy update 
		1. find largest step size that improves the surrogate objective while remaining within the trust region -> adjusts step size until it satisfies trust region constraint
		2. **Proposed Methods -> Solutions**
			1. Single Path Method:
				1. estimation procedure -> collect sequence of states
					1. then simulate policy for some number of timesteps to generate a trajectory
					2. Q(action | state) = policy_old (action | state)
					3. Old Q(action | state) computed at each state-action pair by taking discounted sum of future rewards along the trajectory 
			2. Vine Path Method:
				1. sample state probabilities and simulate the policy to generate trajectories
				2. choose subset of N states along trajectories 
				3. rollout set -> sample K actions 

**Update Policy**
	1. policy updated via update direction and computed step size determined in Line Search
		1. update ensures policy moves in direction that improves objective while respecting the trust region constraint 

**Important Concepts**
1. Trust Region -> allowable distance between old policy and updated policy during update step
	1. constrains magnitude of policy changes to ensure stability and avoid large updates
	2. Ball or Ellipsoid policy parameter space -> controlled by epsilon 
		1. choice of epsilon depends on problem and specified via prior experiences
	3. surrogate loss uses term that bounds the policy update step based on the ratio of new probabilities and old probabilities -> (1 - epsilon, 1 + epsilon) range 
	4. We must minimize the trust region constraint -> updates increment at small magnitudes
	5. Prevent abrupt changes that could lead to instability or suboptimal performance

The TRPO algorithm strikes a balance between exploration and exploitation, aiming to find an optimal policy while ensuring stable and monotonic improvement. By constraining policy updates within a trust region, TRPO provides theoretical guarantees on the performance improvement and maintains a smooth policy update trajectory. It has been widely used in various reinforcement learning domains and has shown promising results in practice.


