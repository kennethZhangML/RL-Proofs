
**Problems:**
- Q-learning overestimates action values 
- instabilities in action selection with overestimation 

**Proposed Methods**
1. Double Q-learning: two value functions learned via assigning each experience randomly to update on of the two value functions -> used to determine the greedy policy 
2. Greedy policy and other to determine its value

![[Pasted image 20230611204736.png]]

**Overoptimization due to Overestimation**
- induce upward bias from noise, function approximation, non-stationarity

**Reduction of Overestimations**
- Overestimation increases the number of actions 
- Double Q-learning reduces overestimations by decomposing max operation in the target into action selection and action evaluation 
- using online network to evaluate greedy policy, and using target network to estimate its value 
- Empirical results show that double Q-learning can be used at scale to successfully reduce overoptimism -> more stable and reliable training 

![[Pasted image 20230612212657.png]]

