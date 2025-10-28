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

Then,
- $$F_{K}$$ is a **layer**
- **a feature** is the output of a layer. Features are passed to the next layer.
- **the model space is parametrized** as: $$\Theta = \Theta_1 \times ... \times \Theta_{K}$$ (product of parameter space of all layers)
- 

### Linear layers

A linear layer:
- **implements a linear operation**
- is defined by:
    - the **dimensions** of its input $$n_{in}$$ and outputs $$n_{out}$$
    - a **weight matrix** $$W \in \mathbb{R}^{n_{out} \times n_{in}}$$ (more on that below)

#### Affine layers

**A linear layer is almost always an afine layer** that performs:

$$x \rightarrow Wx + b$$

Where:
- $$x$$ is an input feature (usually matrix or vector), $$x \in \mathbb{R}^{n_{in}}$$,
- $$W$$ is the weight matrix, $$W \in \mathbb{R}^{n_{out} \times n_{in}}$$ 
- $$b$$ is a **learnable bias parameter** (matrix or vector), $$b \in \mathbb{R}^{n_{out}}$$
- => the goal is to learn $$W$$ and $$b$$.

**Linear layers are fully connected, or dense**: every neuron in the layer is connected to every neuron in the previous and subsequent layers.

#### Weight matrices

$$W$$ is the weight matrix of a linear layer.

- $$W$$ is a **learnable parameter**: the goal is to optimize $$W$$ to minimize the loss function
- $$n^{in}$$ and $$n^{out}$$ are **hyperparameters**: 
    - they are not learned, but defined by the user
    - they define the parametric family of function of the linear layer of $$W$$ (a layer $$L$$ is defined as the parametric family of functions that perform $$\mathbb{R}^{n_{in}} \rightarrow \mathbb{R}^{n_{out}}$$

#### In Pytorch

```python
# Define a linear (= affine ) layer with input dimension 10 and output
dimension 5
linear_layer = nn.Linear(in_features =10, out_features =5)

# Access the weight parameters
weights = linear_layer.weight

# Access the bias parameters
bias = linear_layer.bias

# Define a linear layer (without bias)
linear_layer = nn.Linear(in_features =10, out_features =5, bias = False)
```

### Non-linear layers

A non-linear layer **does not have any learnable parameter**: they will apply the same operation to every component of the input. Some non-linear layers:
- **hyperbolic tangent**
- **sigmoid**
- **ReLU (Rectified Linear Unit)**. $$\for x \in \mathbb{R}$$, it is defined by: $$\text{ReLU}(x) = max(x,0)$$
