**Markov Decision Process**
Theorem: MDP is a well-defined mathematical framework that models decision-making problems.
We wish to prove that the framework is well-defined within the scope of Q-learning.

To prove such, we must establish some properties of an MDP:
1. State Transition Probability:
	1. state transition probability matrix denoted $P$, defines the probability of transitioning from state $s$ to state $s'$ when taking action $a$ 
		1. Each entry $P(s, a, s')$ represents the probability $P(s' | s, a)$ for:
			1. $P(s, a, s') \ge 0, \forall s, a, s'$ 
			2. $\sum P(s, a, s') = 1, \forall s, a$
2. Reward Function:
	1. the reward function denoted $R$, maps state-action pairs to immediate rewards
	2. Each entry $R(s, a)$ represents the immediate reward received when taking action $a$ in $s$
3. Markov Property:
	1. future state depends only on the current state and action and is independent of the past
	2. $P(s', r | s, a, s_0, a_0, ..., s_{t-1}, a_{t-1}) = P(s', r | s, a)$ 
4. Finite State and Action Spaces:
	1. in an MDP, the state space $S$ and action space $A$ are both finite sets

**Conclusion**: the MDP is a well-defined mathematical framework for modelling decision-making problems using properties that capture essential elements of decision making, including transitions, immediate rewards, and the Markov Property. 

-------------------------------------------------------------------------------
**Proof: Value Function for MDPs**
1. **State-Value Function**: expected cumulative reward for being in state $s$ and following a given policy (defined recursively using the Bellman Equation)

	Bellman Equation (State-Value Function)$: V(s) = E[R+ \gamma * V(s')]$
	$R$ is the immediate reward received upon transitioning from state $s$ to the next state and $\gamma$ is the discount factor for determining the importance of future rewards

	Proof: By in Induction
	Base-Case: For terminal state, $s_t$, the value function for immediate reward is:
			$V(s_t) = R(s_t)$ (since there are no future states)

	Inductive Step: Assume the value function above holds for all states $s'$ reachable from state $s$, then we can prove for given state $s$:
	$V(s) = E[R + \gamma * V(s')] = E[R + \gamma * V(s') | s]$ 
	(according to Bellman Equation, and conditioning on current state $s$, respectively)

	Then, by Markov Property, the future state $s'$ depends only on the current state $s$ and action taken by the agent. Thus, $V(s) = E[R + \gamma * V(s') | s, a]$ 

	Therefore, by induction, this equation represents the expected immediate reward plus the expected discounted value of the future state $s'$ given the current state-action pair $(s, a)$. 

2. Action-Value Function: expected cumulative reward for being in state $s$, and taking action $a$, and following a given policy (also defined recursively by Bellman Equation $Q(s, a)$)