# OR-ML-survey
> to be noted
>>  ***quoted from the original resources***  
>>  **to emphasis**  
>> > original context
# End-to-end model:

## Attention:

### Pointer Networks：
Oriol Vinyals et al. 2015.  NIPS, (TSP, convex hall, delaunay triangulation).  

A sequence-to-sequence model with modified attention mechanism where the output is in form of a distribution over all elements of the input. It delivery solutions to a set of problems that the size of output dictionary depends on the input, which is typical in combinatorial optimization problem. The "pointer" in its name refers to a softmax layer generating a probability distribution function on the input from embedding features of each element in input by both encoder and decoder. In the original paper, they conduct experiments on three classical combinatorial problems: convex hall problem (to find a mininal set of nodes to circle all elements on the graph with their connecting links), delaunay triangulation problem (to find the triangulation form of a given set without any node circled by triangles in the segementation), and travelling saleman problem (to find a rount in 2D Euclideam space visiting each node exactly once and return to the original point with shortest distance). The Pointer Network is reported to show improvement on performance on all three problems in small sets compared to previous machine learning methods while fails in large-scale delaunay triangulation problems. More importantly, invalid tours could be produced by Pointer Network model and a beam search algorithm is deployed to only select the valid ones. Generally, it obtains results close to classifical exact and heuristic solvers in TSP, and is considered as a breakthrough in application of machine learning model in an end-to-end approach to combinatorial optimazaion problems especially the TSP.

### Neural Combinatorial Optimization with Reinforcement Learning:
Irwan Bello et al. 2017. ICLR. (TSP, KP).  

The main drawback of the work by Oriol Vinyals is tied to the supervised training approach, since optimal routes are computationally infeasible in most cases in the domain of combinatorial optimization problems due to NP-hard characteristic, or multipel routes are equally optimal. This results to a limited availability of training examples and the its performance generally depends on the quality of labeled data. In respons to this fact, Irwan Bello et at. proposed model designated as Neural Cominatorial Optimization combining neural network and reinforcement learning and reports a significant improvement on performance in Euclideam planar TSP. With a reward function designed on the tour length, the policy-based reinforcement learning components is drived to graduatly optimize the parameter within the pointer network adhering the policy gradient method ***asynchronous advantage actor-critic (A3C)*** through stochastic gradient descend paradigm. The critic compromises three components: two LSTM models where one acts as encoder and the other as process block, and one two-layer neural network with ReLu activation as decoder deepening by an additional glimpse. In addition to the greedy algorithm premarily adopted in the the first version, another two reinformence learning techniqued provide more insight into evaluation of the tour: the first is called sampling where several sampled candidate routes controled by a temperature parameter are drawn from current status and the excat shortest is chosen for the next step; the other is active search to refine the stochastic gradient policy while search on the potential solution to the processed test instance. This model outperforms not only the original Pointer Network but others widly-used solver regards to average route length and running times.  
***glimpses***:
>  to aggregate the contributions of different parts of the input sequence...yields performance gains at an insignificant cost latency.
### Cominatorial Optimization by Graph Pointer Networks and Hierarchical Reinforcement Learning:
Qiang Ma et al. 2019. Columbia. (symmetric TSP, TSP with time window)  

Graph(enhanced) Pointer network; large-scale TSP problem; hierarchical reinforcement learning

**hierarchical reinforcement learning formulation**: 
  lowest layer: a vanilla markov decision process providing latent variables to the next layer
  middel layer (multi-layer structure): a RNN processing both current status and latent viriables from lower layer
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
  
### Reinforcement Learning for Solving the Vehicle Routing Problem
Mohammadreza Nazari et al. 2018. NIPS, (VRP)
> **Policy model** consists of a recurrent neural network decoder coupled with an attention mechanism 

directly use the embedded output instead of the RNN hidden status.
**RNN structure**: since there is no meaning in the order of inputs, the hidden variables is useless in the context of combinatorial optimization. A change in the input may lead to extra cost on conputation for updating. The model consists of two components: the first is ***a set of graph embeddings***, which is in form of 1-layer GCN, but it simply ***utilize the local information of each node, without incorporating adjacency information***; the second is the RNN decoder, where the dynamic static elements is part of the input.  
**Attention mechanism**: the mechanism is similar to one used in *Pointer Networks*
> the embedding and attention mechanism are invariant to the input order. 

**Training model**: policy gradient approach (***use an estimate of the gradient of the expected return with respect to the policy parameters to iteratively improve the policy***) consisting two networks: an actornetwork to predict the distribution over candidates for the next step and a critic network to assess the reward given current status. Two sampling strategies: greedy and beam search, the latter leads to improved prediction at the cost of a slight increase in computation.
masking scheme: superior to classical solver to VRP, including Clarke-Wright savings heuristic (CW), the Sweep heuristic (SW), and Google’s optimization tools (OR-Tools), and is a proper tool for size-varing situation.

## Attention, Learn to Solve Routing Problem!
Wouter Kool et al. 2019 ICLR, (TSP, two variants of VRP), transformer architecture
>The application of Neural Networks (NNs) for optimizing decisions in combinatorial optimization problems dates back to Hopfield & Tank (1985), who applied a Hopfield-network for solving small TSP instances.  

**Encoder**: a framework similiar to Vaswani et al. (2017) ***"attention is all your need"***
**Attention**: the model could be reasonably considered as a Graph Attention Network
The embeddings are updated with a series of attention layers in form of two sublayers for each, and aggregated by mean as the graph embedding. Both node embedding and graph embedding are proceeded to the decoder as input.  
The two sublayers of the attention layer are namely multi-head attention layer (managing message passing between the nodes) and a node-wise fully connected feed-forward layer, in addition with a skip-connection and batch normalization with ReLu activation.  
**Decoder**: A single-head attention mechanism sequentially generates the final probabilities at timestep t from in to n based on the output of encoder and two ending output of the current partial solution. A special architecture called **context embedding** is designed in this model to represent the docoding context.  
**context embedding**:
> The context of the decoder at time t comes from the encoder and the output up to time t,  

that means the graph embedding features, the node features for the first and last nodes in TSP. The new context node embedding in the next layer is computed by the multi-head attention mechanism, and the single query q_(c) is simply computed on context node embedding. The capatibility of the query is computed by all nodes and those can't be visited at current (for TSP it's the visited nodes) are masked.
**REINFORCE with greedy rollout baseline**:
utilize a rollout-baseline similiar to self-critical training ***but with periodic updates of the baseline policy***：
>  b(s) is the cost of a solution from a deterministic greedy rollout of the policy defined by the best model so far
to determine the baseline policy, they keep the greedy rollout policy unchanged within each epoch (a fixed number of steps), and replace the parameter at the end of each epoch if a significant improvement is verified by the t-test.
## GNN: 

### 
