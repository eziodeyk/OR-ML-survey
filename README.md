# OR-ML-survey
# End-to-end model:

## Attention:
### Pointer Networks：
Oriol Vinyals et al. 2015，
A sequence-to-sequence model with modified attention mechanism where the output is in form of a distribution over all elements of the input. It delivery solutions to a set of problems that the size of output dictionary depends on the input, which is typical in combinatorial optimization problem. The "pointer" in its name refers to a softmax layer generating a probability distribution function on the input from embedding features of each element in input by both encoder and decoder. In the original paper, they conduct experiments on three classical combinatorial problems: convex hall problem (to find a mininal set of nodes to circle all elements on the graph with their connecting links), delaunay triangulation problem (to find the triangulation form of a given set without any node circled by triangles in the segementation), and travelling saleman problem (to find a rount in 2D Euclideam space visiting each node exactly once and return to the original point with shortest distance). The Pointer Network is reported to show improvement on performance on all three problems in small sets compared to previous machine learning methods while fails in large-scale delaunay triangulation problems. More importantly, invalid tours could be produced by Pointer Network model and a beam search algorithm is deployed to only select the valid ones. Generally, it matches or even outperforms classifical exact and heuristic solvers in TSP. It's considered as a breakthrough in application of machine learning model in an end-to-end approach to combinatorial optimazaion problems especially the TSP.



## GNN:
### 


### 
