# OR-ML-survey
> to be noted
>>  ***quoted from the original resources***  
>>  **to emphasis**  
>> > original context

(Two classes of combinatorial optimization problem: partition; subset selection.)   
# End-to-end model 
**(also containing deep reinforcement learning models which are not explicitly trained in an exactly end-to-end manner but worth recommending)**

（Deep reinforcement learning = deep learning + reinforcement learning)
## DEEP LEARNING AND REINFORCEMENT LEARNING
## category  sequence-to-sequence:

### RNN
#### Learning Permutations with Sinkhorn Policy Gradient (pending)
Patrick Emami et al. 2018. (sorting, Euclidean TSP, MWM - maximum weight matching)    
GRU + sinkhorn layer with sinkhorn policy gradient.  
model targets on permutation-like problems such as sorting, Eucilidean TSP, MWM, and so on.

### PN
#### Pointer Networks：
Oriol Vinyals et al. 2015. NIPS. (TSP, convex hall, delaunay triangulation).  
github: https://github.com/devsisters/pointer-network-tensorflow    
A sequence-to-sequence model with modified attention mechanism where the output is in form of a distribution over all elements of the input. It delivers solutions to a set of problems that the size of output dictionary depends on the length of input. It is typical in combinatorial optimization problem and the conventional attention mechanism has limit on efficiency in tackling such type of problem. The novelty in this model, the "Pointer", refers to the neural network using attention mechanism with a soft-max layer generating a probability distribution function on the input from hidden status in both encoder and decoder. To demonstrate its performance, they conduct experiments on three classical geometric problems: convex hall problem, Delaunay triangulation problem, and traveling salesman problem. The Pointer Network is reported to show improvement in performance on all three problems in small sets compared to previous machine learning methods while fails in large-scale Delaunay triangulation problems. More importantly, invalid tours could be produced by Pointer Network model and a beam search algorithm is deployed to only select the valid ones. Generally, it obtains results close to classical exact and heuristic solvers in TSP, and is considered as a breakthrough in application of machine learning model in an end-to-end manner to combinatorial optimization problems especially the TSP.

#### Neural Combinatorial Optimization with Reinforcement Learning:
Irwan Bello et al. 2016. ICLR. (TSP, KP).  
github: https://github.com/MichelDeudon/neural-combinatorial-optimization-rl-tensorflow        
The main drawback of the work by Oriol Vinyals is tied to the supervised training approach, since optimal routes are computationally infeasible in most cases in the domain of combinatorial optimization problems due to NP-hard characteristic, or multipel routes are equally optimal. This results to a limited availability of training examples and the its performance generally depends on the quality of labeled data. In respons to this fact, Irwan Bello et at. proposed model designated as Neural Cominatorial Optimization combining neural network and reinforcement learning and reports a significant improvement on performance in Euclideam planar TSP. With a reward function designed on the tour length, the policy-based reinforcement learning components is drived to graduatly optimize the parameter within the pointer network adhering the policy gradient method ***asynchronous advantage actor-critic (A3C)*** through stochastic gradient descend paradigm. The critic compromises three components: two LSTM models where one acts as encoder and the other as process block, and one two-layer neural network with ReLu activation as decoder deepening by an additional glimpse. In addition to the greedy algorithm premarily adopted in the the first version, another two reinformence learning techniqued provide more insight into evaluation of the tour: the first is called sampling where several sampled candidate routes controlled by a temperature parameter are drawn from current status and the excat shortest is chosen for the next step; the other is active search to refine the stochastic gradient policy while search on the potential solution to the processed test instance. This model outperforms not only the original Pointer Network but other widly-used solvers in criteria of average route length and running times.  
***glimpses***:
>  to aggregate the contributions of different parts of the input sequence...yields performance gains at an insignificant cost latency.

#### Cominatorial Optimization by Graph Pointer Networks and Hierarchical Reinforcement Learning:
Qiang Ma et al. 2019. Columbia. (symmetric TSP, TSP with time window)  
github: https://github.com/qiang-ma/graph-pointer-network    
Graph(enhanced) Pointer network; large-scale TSP problem; hierarchical reinforcement learning

**hierarchical reinforcement learning formulation**: 
  lowest layer: a vanilla markov decision process providing latent variables to the next layer
  middle layer (multi-layer structure): a RNN processing both current status and latent viriables from lower layer
  highest layer: only process latent variables from lower layer without providing latent variables.
  
  hierarchical policy gradient:
    **central self-critic** 
```
similar to self-critic baseline and the roll-out baseline in attention model
```  
graph pointer network:
  pointer network(2017) + graph embedding layers;  
  **encoder**:  point encoder（LSTM）and graph encoder (GNN)，the point encoder embeds features on nodes into a high-dimensional space through a weight-sharing linear transformation, and the embedded features along are further encoded by a LSTM model. The output of LSTM, mostly called hidden variables, are passed to not only decoder(attention) along with output from graph embedding layer but encoder(LSTM) in the next recursion.  
  
  **vector context**: on contrast to point context where X stands for the spacial coordinates, use a matrix consisting vectors from one node to another as the input of GNN.
  
  **decoder**: attention mechanism provides a distribution over all remainding nodes via a softmax layer on which model samples for the next candidate.
  
  **finetuned hierarchical GPN**: the lower pointer variable u generated by lower layer LSTM is added to the one from higher layer LSTM in form of the softmax transformation.
  
  performance:   
  **GPN**: the trained model could be well-generalized to large-scale graphs (up to 1000 nodes) when trained on small-scale graphs within less running time; the graph embedding is proved to be useful since the it outperforms the pointer network, but still inferior to graph attention model; the local search algorithm 2-opt is applied to improve the performance in large-scale setting.   
  **Two-layer heirarchical GPN**: add penalty of leaving beyond time constrains to the loss function of lower layer; total time cost as loss function in higher layer plus penalty defined in previous; outperforms all other baselines including OR-tools by Google and heuristic ant colony optimization algorithm.  
  
### Attention
#### Reinforcement Learning for Solving the Vehicle Routing Problem
Mohammadreza Nazari et al. 2018. NIPS, (VRP)  
github: https://github.com/OptMLGroup/VRP-RL   
> "dropped the original RNN part of the encoder model completely and replaced it with a linear embedding layer with shared parameters" -- Jonas K. Falkner et al. 2020   

> **Policy model** consists of a recurrent neural network decoder coupled with an attention mechanism 

directly use the embedded output instead of the RNN hidden status   
**RNN structure**: since there is no meaning in the order of inputs, the hidden variables is useless in the context of combinatorial optimization. A change in the input may lead to extra cost on conputation for updating. The model consists of two components: the first is ***a set of graph embeddings***, which is in form of 1-layer GCN, but it simply ***utilize the local information of each node, without incorporating adjacency information***; the second is the RNN decoder, where the dynamic static elements is part of the input.  
**Attention mechanism**: the mechanism is similar to one used in *Pointer Networks*
> the embedding and attention mechanism are invariant to the input order. 

**Training model**: policy gradient approach (***use an estimate of the gradient of the expected return with respect to the policy parameters to iteratively improve the policy***) consisting two networks: an actor network to predict the distribution over candidates for the next step and a critic network to assess the reward given current status. Two sampling strategies: greedy and beam search, the latter leads to improved prediction at the cost of a slight increase in computation.  
masking scheme: superior to classical solver to VRP, including Clarke-Wright savings heuristic (CW), the Sweep heuristic (SW), and Google’s optimization tools (OR-Tools), and is a proper tool for size-varing situation.  

#### Learning Improvement Heuristics for Solving Routing Problems (pending)
Yaoxin Wu et al. 2020. RL with attention-based policy network + 2-opt. (TSP + CVRP).   
github: https://github.com/yining043/TSP-improve   

### Transformer
#### Learning Heuristics for the TSP by Policy Gradient
Michel Deudon et al. 2018, CPAIOR, (TSP)  
github: https://github.com/MichelDeudon/encode-attend-navigate   
The authors proposed a model originated from the Pointer Net by Ballo(2016) soly based on attention mechanisms (self-attention, Vaswani 2017) instead of the LSTM architecture.
**The extended neural combinatorial optimization framework**:

**Encoder**:  
the encoder maps the input set into a latent space, by which the decoder generates the stepwise output in the auto-regressive manner.  
To address orientation issue, they firstly centered and then applied PCA over all nodes. After embedding and batch-normalization, the processed features are led to the N-layer encoder where each layer consists two sublayer -- the multi-head attention and the feed-forward (***two position-wise linear trans- formations with a ReLU activation in between***), and the output is in form of:  
> LayerNorm(x + Sublayer(x))   
**the multi-head attention**:  
linear transformation for h times in parallel of Q, K, V into lower dimensionals vector and feed them forward to the attention mechanism. 

**Decoder**:  
> explicitly forgets after K = 3 steps, dispensing with LSTM networks.  
say, maps query at time t to the last three actions. 

**Training**:  
> trained by Policy Gradient using the REINFORCE learning rule with a critic to reduce the variance of the gradients.  

**policy gradient and REINFORCE**.   
key words: a critic network to reduce the variance of the gradients. And Monte-Carlo sampling with to approximate the gradient in batch;  
the critic uses the glimpse technic ***computed as a weighted sum of the action vectors***.  
>  fed to a 2 fully connected layers with ReLu acti- vations. The critic is trained by minimizing the Mean Square Error between its predictions and the actor’s rewards.

**performance**:
outperforms the pointer net in small-scale cases but doesn't keep advantage in large-scale cases.


#### Attention, Learn to Solve Routing Problem!
Wouter Kool et al. 2019 ICLR, (TSP, orienteering problem, prize collecting TSP, stochastic PCTSP, ), transformer architecture   
github: https://github.com/wouterkool/attention-learn-to-route    
> "replaced...with an adapted transformer model employing self-attention [28] instead." -- Jonas K. Falkner et al. 2020

>The application of Neural Networks (NNs) for optimizing decisions in combinatorial optimization problems dates back to Hopfield & Tank (1985), who applied a Hopfield-network for solving small TSP instances.  

**Encoder**: a framework similiar to Vaswani et al. (2017) ***"attention is all your need"***   
**Attention**: the model could be reasonably considered as a Graph Attention Network   
The embeddings are updated with a series of attention layers in form of two sublayers for each, and aggregated by mean as the graph embedding. Both node embedding and graph embedding are proceeded to the decoder as input.   
The two sublayers of the attention layer are namely multi-head attention layer (managing message passing between the nodes) and a node-wise fully connected feed-forward layer, in addition with a skip-connection and batch normalization with ReLu activation.   
**Decoder**: A single-head attention mechanism sequentially generates the final probabilities at timestep t from in to n based on the output of encoder and two ending output of the current partial solution. A special architecture called **context embedding** is designed in this model to represent the docoding context.      
**context embedding**:   
> The context of the decoder at time t comes from the encoder and the output up to time t,

that means the graph embedding features, the node features for the first and last nodes in TSP.   
The new context node embedding in the next layer is computed by the multi-head attention mechanism, and the single query q_(c) is simply computed on context node embedding. The capatibility of the query is computed by all nodes and those can't be visited at current (for TSP it's the visited nodes) are masked.   
**REINFORCE with greedy rollout baseline**:  
utilize a rollout-baseline similiar to self-critical training ***but with periodic updates of the baseline policy***：  
>  b(s) is the cost of a solution from a deterministic greedy rollout of the policy defined by the best model so far  
to determine the baseline policy, they keep the greedy rollout policy unchanged within each epoch (a fixed number of steps), and replace the parameter at the end of each epoch if a significant improvement is verified by the t-test. 

(dont' get it clearly)  
**performance**: less optimal than the state-of-the-art solvers such as LKH3 and Gurobi but show advantage in convergence speed.

#### A Deep Reinforcement Learning Algorithm Using Dynamic Attention Model for Vehicle Routing Problems (pending)
Bo Peng et al. 2020, (SYSU), (VRP)  
**dynamic attention model (AM-D)**   
encoder-decoder architecture in the schema of attention with mask.   
Two strategies: sample rollout and greedy rollout

#### Learning to Solve Vehicle Routing Problems with Time Windows through Joint Attention (in process)    
Jonas K. Falkner et al, 2020, (CVRP, CVRP-TW)    
**encoder-decoder architecture with multi-head attention mechanism**  
**encoder**: nodewise input includes coordinates and demands; ***"consisting of a stack of three self-attention blocks"***, each block follows form of MHA --> BN --> FF with res --> BN.  
A multi-head attention layer is a linear combination of  several single-head attention layers, each takes disjoint part of the input.
A single-head attention layer is ***“a convex combination of linearly transformed（weighted softmax) elements with parameterized pairwise attention weights”***
**Decoder**: an attention layer with multi-head attention inputs and masked sequence.
**JAMPR**: node encoder as described above in addition with tour encoder and vehicle encoder, all together provide context embeddings (graph, fleet, act, node, and last) and sequence embedding (vehicle and nodes)

## category graph:  
### embedding:   
#### Learning Combinatorial Optimization Algorithms over Graph.   
Hanjun Dai et al. 2017, nips, TSP  
github: https://github.com/Hanjun-Dai/graph_comb_opt   
This is a work of connerstone for the art of introducing the graph embedding and deep reinforcement learning into the domain of the combinatorial optimization and a benchmark for later researches. The authors introduce a graph embedding network called structure2vec to abstract information on the graph structure and node covariates into embedding feature vectors. These feature vectors are transmitted to the approximated evaluation function. A reinforcement learning method called Q-learning is deemed as the natural choice to parameterize the Q function.  
**structure-to-vec**:  
> the computation graph of structure2vec is inspired by graphical model inference algorithms, where node-specific tags or features x are aggregated recursively according to G’s graph topology. The node specific features as well as long-distance interaction is considered in the model through reccursive computation.

parameterization: embed three parts of information including the binary scalar (or other useful node information), the neighborhood embedding, and weights of adjacent links; then feed output into the Q function.  
**Q-learning**:   
policy: a deterministic greedy policy is to select the node on the remaining part of the graph that results to cellect reward.  
learning algorithm:   
> a combination of n-step Q-learning (off policy, value-based, help to deal with delayed rewards, the procedure of waiting n steps before updating the approximator's -- Q-hat's -- parameters ) and fitted Q-iteration (uses experience replay; is shown to result in faster learning convergence when using a neural network as a function approximator; stochastic gradient descent updates are performed on a randome sample drawn from E ).  

summary: n-step Q-learning to alleviate the curse of being ***myopic*** with greedy policy; fitted Q-iteration for ***fast learning convergence***; supervised learning method for sample efficiency.
**training and experimental performance**:  
> The hyperparameters are selected via preliminary results on small graphs, and then fixed for large ones; train the model on graphs with up to 500 nodes.  
in comparison to pointer network with actor-critic algorithm:     
> To handle different graph sizes, we use a singular value decomposition (SVD) to obtain a rank-8 approximation for the adjacency matrix, and use the low-rank embeddings as inputs to the pointer network.  

the model trained on small-scale graphs could be well generalized to large-scale ones with up to 1k nodes.  

### Message Passing Neural Networks
#### Neural Message Passing for Quantum Chemistry (pending)
Justin Gilmer et al. 2017, supervised learning,  
github: https://github.com/microsoft/tf-gnn-samples   

#### Learning a SAT Solver from Single-bit Supervision (in process)
Daniel Selsam et al. 2019, ICLR, supervised learing, (SAT).   
SAT: A formula of propositional logic wih n variables and m clauses is said to be satifiable if ***there exists an assignment of boolean values to its variables such that the formula evaluates to 1***.   
> A SAT problem is a formula in CNF, where the goal is to determine if the formula is satisfiable, and if so, to produce a satisfying assignment of truth values to variables.  

Given a SAT problem, the main goal is to approxiate an sufficient and necessary fomular to it, and ultimately to solve task (***"finding solutions to satisfiable problems"***).

***NeuroSAT***:(an end-to-end SAT solver with good scalability):
> parameterized by two initial vectors, three MLPs, and two layer-norm LSTMs.

#### Exploratory Combinatorial Optimization with Reinforcement Learning   
Thomas D. Barrett et al. 2020, reinforcement learning, (max-cut)  
github: https://github.com/tomdbar/eco-dqn  
Thomas D. Barrett et al. 2020, reinforcement learning, (max-cut)   
ECO-DQN: exploratory combinatorial optimization - Deep Q-Network.  

* Q-learning:  
**policy**: a map from a state to a distribution probability over action.  
**value function**: given a state-action pair, its Q-value is ***the discounted sum of immediate and future rewards*** starting from the initial state & action and following the optimal policy.   
**deep Q-network**: a function with parameters to approximate the maximum of Q-value following the optimal policy. **Once trained, an approximation of the optimal policy can be obtained simply by acting greedily with respect to the predicted Q-values**.  
***explore the solution space at test time, rather than producing only a single “best-guess”***
**message passing neural network**: use the MPNN as the deep Q-network in the model -- ***a general framework of which many common graph networks are specific implementations***.  
It embedds each vertex in the graph into a n-dimensional features in the following way: the input vector of observation is firstly initialised by the a function and then iteratively updated one node at a time with information from neighbors through message and update functions.  

**exploiting exploration (yet unclear: it feels like )**: ***trained to explore the solution space at test time** (???).   
***the Q-value of either adding or removing a vertex from the solution is continually re-evaluated in the context of the episode’s history. Additionally, as all actions can be reversed***

* reward shaping:   
** to find the best solution with highest cut-value without punishment when choose an action reducing the cut-value.  
** normalised by the total number of vertices.  
* observations:
seven observations are used in Q-value to flip (?) each vertex.
> ECO-DQN ≡ S2V-DQN+RevAct+ObsTun+IntRew

### GNN
#### Revised Note on Learning Algorithms for Quadratic Assignment with Graph Neural Networks  
Alex Nowak  et al. 2017, PMLR, (graph matching, TSP)  
github: https://github.com/alexnowakvila/QAP_pt    
In this work, researchers try to directly use a ***siamese*** GNN to process two given graphs into embedding features and predict matching basd on these features.
(It states that there are two approaches to train models: ground-truth based and cost based.) The author stated promising results through supervised learning, and its performance on TSP is slightly less optimal than Pointer Network. The main drawback of this model is ***the need for expensive ground truth examples*** and the gap to the optimal solver is hypothetically due to the model architecture.


#### Learning to Solve NP-Complete Problems: A Graph Neural Network for Decision TSP
Marcelo Prates et al. 2019, AAAI, (TSP)   
github: https://github.com/machine-reasoning-ufrgs/TSP-GNN     
The decision TSP refers the question on whether there exists the solution of cost under a certain criteria.
***The GNN model for TSP***:
the role of the graph neural network is divided into two parts: the first one is to ***assgin a multidimensional embedding to each vertex***; and the second is to ***perform a given number of message-passing iterations*** (where the embedding of each node is transmitted to its adjacencies as their incoming messages). Those incoming messages are added up and fed into a RNN. ***The only trainable parameters of such a model are the message computing modules and the RNN***.   
**For the TSP**
**additionally assign embedding to edges** for the information about edge weights, replace the vertex-to-vertex adjacency matrix with an edge-to-vertex matrix adjacency matrix connecting each edge to its source and target vertices; the given target cost C together with the weight of the edge are concatenated and fed into a multi-layer perceptron to be expended into the initial embedding for that edge after a given number of iterations.   
>finally the refined edge embeddings are fed into an MLP which computes a logit probability corresponding to the model’s prediction of the answer to the decision problem

Training Target: to minimize the binary cross entropy lost between the prediction and the ground-truth.
**performance**:???

#### Deep Reinforcement Learning meets Graph Neural Networks: exploring a routing optimization use case   
Paul Almasan et al. 2020, DRL+GNN, (optical transport networks: ***as a classical resource allocation problem in network***)  
github: https://github.com/knowledgedefinednetworking    
A SDN-based Optical Transport Network under Deep Q-learing framework: ***"use a DNN as the q-value function estimator"***.   
The DRL agent follows DQN algorithm where a DQN models the q-value function in message passing architecture:   
> an iterative message passing process runs between the link hidden states according to the graph structure    
Evaluation: compared to other state-of-the-art models (two kinds of graph topology -- the 14-node NSFNet topology and the 24-node Geant2 : trained on one of them and tested on the other one) including ***a state-of-the-art DRL-based solution, a load balancing routing policy, and a theoretical fluid model***.    

#### Solving NP-hard Problems on Graphs with Extended Alpha-Zero (in process) 
Kenshin Abe et al. 2020, RIKEN.  
github: https://github.com/xuzijian629/combopt-zero     
MCTS to train the deep neural network; combined with Graph Neural Network (including 2-IGN+, GIN, GCN, S2V) in the second phase.   
**algorithm**:   
MCTS search tree: similiar to MCTS in addition with **the mean and the standard deviation of the results by random plays** stored in nodes.   
MCTS: each iteration contains three parts -- select, expand, and backup...   


#### Graph Colouring Meets Deep Learning- Effective Graph Neural Network Models for Combinatorial Problems (pending)
Henrique Lemos et al. 2019, (graph coloring)

#### Fast Detection of Maximum Common Subgraph via Deep Q-Learning:
Yunsheng Bai et al.  2020, Deep Q-learning, (Maximum Common Subbgraph -- MCS)   
Background:  exact MCS solvers do not have worst-case time complexity guarantee and cannot handle large graphs in practice   
RLMCS: a graph neural network (joint subgraph-node embedding) based Deep Q-learning Model.  
In addition with a Fast Iterative Graph Isomorphism algorithm for graph isomorphism checking.

**Policy Network**:   
* consisting of a graph embedding network and a deep Q-network (mainNet for current Q-value, targetNet for target Q-value, experience replay ***to alleviate the problems of correlated data and non-stationary distributions***, update the targetNet by copying from the mainNet every N timesteps)    
* joint subgraph-node embedding for embedding generation: ***produces the embeddings of all the nodes and the subgraph jointly***

**Action Prediction**:   
concatenate node embeddings, subgraph embeddings, and graph-level embeddings generated by JSNE as input into a multi-layer to predict Q-values

**Subgraph Exploration Tree** inspired by beam search, use a beam size to guide search from current status.

**performance**: compared with other three groups, namely exact solvers, supervised models, and unsupervised models.


### GCN
#### An efficient graph convolutional network technique for the travelling salesman problem
Chaitanya K. Joshi et al. 2019, supervised learning.   
github: https://github.com/chaitjo/graph-convnet-tsp.  
keywords: non-autoregressive -- ***"outperforms all recently proposed autoregressive deep learning techniques"***.  
The model takes the whole graph as input and proceed it into an adjancent matrix as the solution to the TSP. Given the ground-truth solution to graph at hand, they employed the supervised learning method by defining the loss function based on batch-averaged ***weighted binary cross-entropy***. It computes h-dimensional vectors for each node and edge on the graph, and the final edge features are projected through softmax layer into the ground-truth TSP tour.   
The main point of this work locates itself in the comprensive description on the structure of its GCN model, which consists of two types of layers: the input layer and convolutional layer.  
**input layer**: the two-dimensional coordinates are embedded into h-dimensional vectors, the Euclidean distances of edges concated with three indicies (1 for k-nearest neighbor, 2 for self-connection, and 0 for others) to h/2-dimensional ones. It states that the input of k-nearest neighbors may contribute to speed up the convergence in TSP.  
**graph convolutional layer**:  
the graph convolutional layers works on both node-embedding features and edge-embedding features and hierachically proceeds them into the final MLP layer, which generates the adjancent matrix as output.   

#### End to end learning and optimization on graphs 
Bryan Wilder et al. 2019 nips, differentible approximation.   
github: https://github.com/bwilder0/clusternet.  
> include more structure via a differentiable k-means layer instead of using more generic tools (e.g., feed-forward or attention layers)...use a differentiable approximation to the objective which removes the need for a policy gradient estimator.   
In most common approaches, model is responsible to rebuild the unknown ground-truth adjancent matrix from known ones training data by minimizing the handcrafted loss function. By constrast, the paper presents an end-to-end model directly mapping the known adjancent matrix to a ***feasible*** decision.  

The model focused on the combination of leanrning task and optimization task where the input graph is partially observed. The learning task is to learn the underground truth of graphs through inputs and the optimization task is to solve the optimization problems based on the predicted graph. The model follows the end-to-end architecture that directly maps the partial observations to the decision by maximizing the decision quality.

Model consists of two components: one is the GCN model aggregating node features and graph structures into multi-dimensional vectors, and a differentiable optimization takes the embedded vectors into solutions.

Two pipages for the differentiable k-means: forward and backward for the clustering procedure 
**Forward pass**:  
***run a solver***   
> "a soft-min assignment of each point to the cluster centers based 􏰁on distance".  
and iteratively update the cluster centers until they converge to fixed points.
**Backward pass**:  
***sensitivity analysis via KKT condition***
Two approaches: exact backward pass and approximate backward pass.   

Experimental Part:  
Two problems: learning(***observe a partial graph and aim to infer which unobserved edges are present***), optimization(***community detection, facility location***). 


#### Learning Heuristics over Large Graphs via Deep Reinforcement Learning
Sahil Manchanda et al. 2020 (under preview), (influence maximization --  budget constraints)  
The scope of further development on S2V-DQN and GCN-TREESEARCH:
* scalability
* generalizability to real-life combinatorial problems
* budget constraints   
**GCOMB**: consisting of two components: ***a GCN to prune poor nodes and learn embeddings of good nodes in a supervised manner***, and a DQN to predict solutions on the good nodes.
> a novel probabilistic greedy mechanism...choose a node with probability proportional to its marginal gain  
**probabilistic greedy**: to generate training samples for GCN from the solution space by ***choose a node with probability proportional to its marginal gain*** under termination condition with a minimum marginal gain contribution, which is as well learned by the embeddings from GCN.
Two-fold destination of the GCN component: learn to identify potential noisy node on graph (noise detector) and to represent the quality of good nodes (node quality predictor under budget constraints).

#### Learning 2-opt Heuristics for the Traveling Salesman Problem via Deep Reinforcement Learning
Paulo Roberto et al. 2020, (TSP)  
github: https://github.com/paulorocosta/learning-2opt-drl   
**"Policy Gradient Neural Architecture**:A policy neural network with a pointing attention mechanism  

**Reinforcement Learning Formulation**:
the state is compused of the current solution and the best solution ever seen before.    
the action corresponds to the 2-opt move.
two networks: one for the policy and the other for the value.

**Encoder**:   
> encode edge information using graph convolutions and use classical sequence encoding to learn tour representations.  

element embedding from Graph Convolutional Network and sequence embedding from Recurrent Neural Network -- embedding layer copes with two dimensional coordinates of nodes, and then transmits them into GCN layer, which ***"leverages node features with the additional edge feature representation"***; 

  **Sequence Embedding Layers**   
Two LSTM models (one in forward manner and one in backward) process the output from GCN model as features in **each** layer and colletively form ***unique node representations in a tour***;  

  **Dual Encoding Mechanism**:  
Each status S_t bar is in form of tuple consisting of the current route as well as the best solution ever encountered with minimum cost (NB: current solution may not follow the minimum-cost criteria).  

**Decoders**: policy decoder and value decoder.   
> decode these repre- sentations via a pointing attention mechanism.  

  **policy decoder**: 
> "given a status...assigns high probabilities to moves that reduce the cost of a tour."   

They use ***"individual softmax functions to represent each*** multiplicative term in the factorized probability of a k-opt move. The query vector at one step is reccursively obtained additionally with the sequence output. And ultimately a Pointing mechanism is  to ***predict a distribution over node outputs given encoded actions (nodes) and a state representation (query vector)***.   

**value decoder**:  
> "reading tour repre- sentations from S and S′ and a graph representation from S"

**policy gradient optimization**:
???

### category others：
#### Routing in Optical Transport Networks with Deep Reinforcement Learning
JOSÉ SUÁREZ-VARELA et al. 2020. HUAWEI. DRL

#### Erdos Goes Neural: an Unsupervised Learning Framework for Combinatorial Optimization on Graphs
Nikolaos Karalias et al. 2020. NIPS, (the maximum clique, the constrained minimum cut)

## Traditional Machine Learning Models as end-to-end methods
#### Targeted sampling of enlarged neighborhood via Monte Carlo tree search for TSP    
Zhang-Hua Fu et al. 2020. ICLR. TSP.   
github: https://github.com/Spider-scnu/Monte-Carlo-tree-search-for-TSP   

# Machine Learning & Exact methods.    



#### Learning to Search in Branch-and-Bound Algorithms
He He et al. 2014. nips. linear classifier.
notes:    
* ***no-problem-dependent learning: use imitation learning to automatically learn the heuristics***.   
* ***aim to search for a good solution without a rigorous proof of optimality.    

two policies:
* ***node selection policy determines the priorities used......obtain...by learning a linear rank function***, the node with highest priority will be the object of the node pruning process.    
* ***node pruning policy decides whether to prune or expand given the current partial progress of the solver.......a binary classifier/regressor takes the feature-vector description of the state and attemps to predict the oracle action****    

the **orcacle**: ***expend nodes in an order that minimizes the number of node expansions subject to finding the optimal solution.***    

two heuristic methods: DFS and best-found-first search.   
**the DAgger learning algorithm**:    
* ***an interative imitation learning algorithm***.   
* use the LIBLINEAR as the classifier to learn the policies.    

performance: the model consistently performs well on all four datasets including MIK, Regions, Hybrid, CORLAT compared to other two cutting-edge open-source solvers --  SCIP and Gorubi.

#### A Supervised Machine Learning Approach to Variable Branching in Branch-And-Bound    
Alejandro Marcos Alvarez et al. 2014. Extremely Randomized Trees.    
to imitate the strategy of strong branching with the machine learning model, namely extremely randomized trees, ***with a fast approximation... in an off-line fashion*** to alleviate the computational cost in exhaustive branching.   
**two-phased approach**: firstly generate the heuristic decision to be learned by machine learning agent, then train the model with generated dataset and examine its performance on standard benchmark problems.

The functional form of the branching is to select nodes that recurssively maximizing the specific branching score. Such branching score function is simulated by the supervised machine learning model with hand-designed input features. These features process three properties:
> 1. the number of features is independent of the size of the problem.    
> 2. invariant to irrelavent changes in the problem.    
> 3. independent of the scale of the problem.   
and are divided into three categories: static problem features, dynamic problem features, and dynamic optimization features.

**performance:**
comparison to other heuristic methods: random, MIB, NCB, FSB, and RB.
Performance on randomly generated examples shows acceptable gap to the optimal solver (full strong branching) with significant reduction in computation.

#### Machine Learning to Balance the Load in Parallel Branch-and-Bound (pending)

#### Learning to Branch in Mixed Integer Programming
Elias B. Khalil et al. 2014. nips. SVM-rank
***a on-the-fly, data-driven design of variable selection strategy***:    
> using a small number of search nodes to approach the good performance of SB    
> maintaining a low computation footprint    
> selecting variables adaptively based on the properties of the given instance.

Three-phased framework: 
* 1. data collection (compute SB scores for every node and ***assign labels to the candidate variables*** => a training dataset.    
* 2. model learning (a learning-to-rank algorithm to output ***the vector of weights for the features***).   
* 3. ML-based branching ( ***branching on the one with maximum score until termination***)    

with three highlighted properties: node-efficiency, time-efficiency, and adaptiveness.    
The training data consists of four components: a set of search tree nodes, a set of candidate variables, labels, and a feature map from variables to features.    
The work ***focuses on learning variable selection strategy***. The model is first instantiated by the CPLEX, then candidate features in both static and dynamic manners is calculated for each node, and lastly a SVM-model -- SVM-rank -- learns ***to estimate a two-level ranking of good and bad variables based on SB scores***.    

The SVM model (or in general a linear classifier) follows the pairwise ranking formulation and train on one pair of nodes at a time. At each new node, the feature vector is computated for each variable and the node is branched on the variables with maximum score. The model outperforms other classical branching strategies in all three considered aspects: the number of unsolved instances, the number of nodes to solve the instances, and the total time to solve the instance.

#### Online Learning for Strong Branching Approximation in Branch-and-Bound (pending)     
Alejandro Marcos Alvarez et al. 2016. Variable Branching.    
online learning

#### A Machine Learning-Based Approximation of Strong Branching (pending)
Alejandro Marcos Alvarez et al. 2017. Extremely Randomized Tree.    
The strong branching strategy is the most efficient one up to now considering the number of explored nodes, while the computation might be considerable costing. The model employs a machine learning method to efficiently imitate the strong branching strategy and alleviate the computational burden.    
Three properties ought to be possessed by the features:    
* ***number of features need to be independent of the size of problem instances***.    
* ......

three types of features: static problem features, dynamic problem features, and dynamic optimization features.

#### Learning a Classification of Mixed-Integer Quadratic Programming Problems     
Pierre Bonam et al. 2017.    
Mixed-Integer Noliear Programming (MINLP) -> Mixed-integer Quadratic Programming (MIQP).   
four tested models: support vector machine with RBF kernel, Random Forests, Extremely Randomized Trees, Gradient Tree Boosting.    
Optimization in quadratic form: Q is a symmetric matrix; a MIQP relaxation drops the integer constrains away.    
Three problem types: pure convex, pure convex, and pure non-convex
   
The ***offline and supervised nature*** facilitates the task to examine influence of centain features to the decision-making.

Four main phases: dataset generation -> features design -> labels definition -> learning experiments. 
Methodological details:    
* dataset generation: constraint set includes cardinality, simplex, and multi-dimensional knapsack
* feature design: groups -- static features, dynamic features
* labels definition: linear, non-linear, and tie (the probability of a tie between the two methods); three checks -- solvability, seed consistency, and global consistency.    
Result: accuracy, precision, recall, and f1-score.    
further work: analyse other benchmark datasets; identify the best learning scenario; define a custom loss function.

#### Learning When to Use a Decomposition
Markus Kruber et al. 2017.    
core idea: ***Given a MIP, a DW decomposition, and a time limit, will GCG using DW optimally solve faster than SCIP?***.     
note: GCN -- ***an extention to well-established MIP solver SCIP***; ***several detectors look for possible DW reformulation of the original MIP model***;
The key factors: 1. the features catch relevant information; 2. training data represent the test data; 3. use a appropriate binary classifier (nearest neighbors; support vector machine with an RBF kernel, random forests).

features for the binary classifier: instance statistics (decomposition-based statistics); detectors; time-remaining after detection.

comparison among four solvers: ***the standard branch-and-cut MIP solver SCIP; GCG that tries to detect a decomposition and perform a DW reformulation accordingly***; SL stands for the methodology proposed in the paper; OPT by alwayls selecting the best solver.

Performance:     
> ***SL scheme with KNN catches roughly 2/3 of the improvement potential of OPT with respect to SCIP***.   
> accuracy at up to 80 percents on subset of cases where GCG is selected by the model.    


# Machine Learning & Heuristic 

# Others    
#### DASH: Dynamic Approach for Swichting Heuristics     
Branching Heuristics:      
* Most Fractional Rounding (MF)    
* Less Fractional Rounding (LF)    
* Less Fractional and Highest Objective Rounding (LFHO)    
* Most Fractional and Highest Objedtive Rounding (MFHO)     
* Pseudocost Branching Weighted Score (PW)   
* Pseudocost Branching Product Score (P)

Feature Space:    
* Percentage of variables in the subproblem;    
* Percentage of variables in the objective function of the subproblem;      
* Percentage of equality and inequality constraints;      
* Statistics of how many variables are in the constraint;    
* Statistics of the number of constraints in which each variable is used;    
* Depth in the branch and bound tree.    

Solvers: BSS; RAND 1(***switch between heuristics uniformaly at random***); RAND 2(***switch among the top 4 best heuristics***); VBS; VBS_band.    
