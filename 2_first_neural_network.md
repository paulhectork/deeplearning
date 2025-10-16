# First neural network

## MLP: Multi-layer perceptron

(and not my little poney)

A parametric function may be defined from **the composition of simpler paramatric functions** ([composition](https://en.wikipedia.org/wiki/Function_composition)). 

Having:
- $$K$$ families of simple parametric functions $$F_{1}, ..., F_{K}$$ 
- $$\Theta_{1}, ..., \Theta_{K}$$ the parameter space of each family $$F_{1}, ..., F_{K}$$
- for $$k \in {1, ..., K-1}$$, the ouput space $$Y_k$$ of a family $$F_{k}$$ is the same as the input space $$X_{k+1}$$ of $$F_{k+1}$$ 
    - in short, $$Y_k = X_{k+1}$$
    - (in other words, the output of the functions in a family can be piped to the input of the functions in the next family)

For $$k \in {1, ..., K}$$, a family of functions $$F_k$$ can be written as: $$F_k = \\{ f_{\theta}: X_k \rightarrow Y_k | \theta \in \Theta_{k} \\}$$
- in other words, $$F_k$$ is 
    - the family of all functions that performs a transformation $$X_k \rightarrow Y_k$$
    - for all parameters $$\theta \in \Theta_k$$
- since the $$Y_k = X_{k-1}$$, we can say that $$F_k = \\{ f_{\theta}: X_k \rightarrow X_{k+1} | \theta \in \Theta_{k} \\}$$

A model space $$F$$ can be defined as **the space of the composite functions**:

$$F = \\{ f_{K} \circ ... \circ f_{1} | f_{1} \in F_{1}, ..., F_{K} \\}$$


