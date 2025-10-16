# ML Basics

## Notations 

(notations are different in the `Supervised parametric learning` part)

- input space: $$X = \mathbb{R}^{in}$$ and output space $$X = \mathbb{R}^{out}$$
- model space: $$F = {f_{\theta}: X -> Y | \theta \in \Theta}$$
- training set: $$D$$, validation set $$D_{val}$$, test set $$D_{set}$$
- cost function: $$l(y´,y)$$ for a prediction $$y' \in Y$$ and a target $$y \in Y$$
- loss function: $$L(\theta,D)$$ for parameters $$\theta \in \Theta$$ and dataset $$D$$
- training samples: $$D = \\{(x_{1}, y_{1}), ..., (x_{N}, y_{N})\\}$$ with number of training samples (data points): $$N = |D |$$

## Intro

Contrary to most scientific methods (i.e., physics), ML isn't about building a mathematical model of reality. It is about being able to predict an outcome based on a set of data points, without formalizing a model of reality. 

Different learning scenarios:
- **supervised learning**: in your dataset, you include examples of desired outputs for your model. 
    - tasks: classification, regression...
    - => generalisation issues
- **unsupervised learning**: the input is raw data points without desired outputs.
    - tasks: clustering, density estimation, outlier detection
- **other**: semi-supervised, self-supervised...

Different ML techniques:
- **parametric**: uses parametric functions (with defined parameters). The aim is to **optimise parameters** to fit an objective function. ex: Neural networks
- **non-parametric**: the model depends directly on the data, without optimising parameters. ex: nearest-neighbour algorithm

## Supervised parametric learning

A supervised algorithm takes an input space $$X$$ (= $$\mathbb{R}^{n}$$) and **makes a prediction** in space $$Y$$ (= $$\mathbb{R}^{m}$$). $$\mathbb{R}^{n}$$ and $$\mathbb{R}^{m}$$ are vector spaces with $$n \in \mathbb{R}$$, $$m \in \mathbb{R}$$, $$n > 0$$, $$m > 0$$.

A model is defined in a **model space** $$F = \\{f_{\theta} : X \rightarrow Y|\theta \in \Theta\\}$$
- the model space is a **function space**, that is a family of all functions that transform input space $$X$$ to $$Y$$.
- $$f$$ are parametric functions
- $$\Theta$$ is the set of all possible parameters. $$\theta$$ is an instance (a specific set of parameters)
- the goal is to **optimise the parameters $$\theta$$** to find the best predictor function $$f$$.

Parameters are also called **weights**.

### Risk and loss

We have 
- $$f$$ a prediction function defined for a single set of parameters $$\theta \in \Theta$$. 
- $$f(x)$$ is an application of $$f$$ 
- $$x$$ a single set of data points.

How do we measure the efficiency of $$f$$ ?

Risk and loss are **formalizations** of the idea of *best prediction*. In short, 
- for a single set of parameters, there is **1 risk, several losses**
- **cost** measures the quality of a single prediciton for a single set of data points
- **risk** is the average of all losses for all data points for this set of parameters.

#### **$$l(f(x),y)$$ - cost function**

- signature:  $$l: Y \times Y \rightarrow \mathbb{R}$$.
- it can be expressed as: $$l(y',y)$$, with 
    - $$y$$: the expected output
    - $$y'$$: the actual output of a prediction $$f$$
    - in other words, $$l$$ compares actual output of $$f$$ to expected output $$y$$

#### **$$R$$ - Risk = average loss over all data points**
- in **statistical learning**, $$R(f) = \mathbb{E}_{(x,y) \sim Z}[l(f(x),y)]$$. 
    - we average losses for all data points $$(x,y)$$ randomly sampled from a distribution $$Z$$.
    - $$Z$$ is a probability distribution made of tuples $$(x,y)$$ where $$x$$ is the input data points and $$y$$ is the output data points.
    - in practice, the entire $$Z$$ cannot be known (we cannot know the entierty of the dataset a model will ever be used on) => we need another measure
- since $$Z$$ can never be known, we **define $$Z_{2}$$**, a known subset of $$Z$$ such that $$(x,y) \in Z_{2}$$ and $$Z_{2} \in Z$$.
- **empirical risk** is a measure of risk where all values of $$Z_{2}$$ are known:
    - $$R_{Z_{2}}(f) = \frac{1}{|Z_{2}|} \sum_{z \in Z_{2}}^{} l(f(x),y)$$
    - $$Z_{2}$$ is a known set of samples $$(x,y) \in Z$$ => $$Z$$ is our dataset and a subset of all possible occurrences $$Z$$. 

**$$L(\theta, Z_{2})$$ - loss function = cost for all data points for a single set of parameters \theta**
- it is expressed as: $$L(\theta, Z_{2})$$
- our goal is to **minimize the loss function $$L$$ for a single \theta.

### Train, validation, test data

To avoid overfitting, in supervised learning, $$Z_{2}$$ is split into:
- $$Z_{2train}$$: training dataset, used to train the model
- $$Z_{2val}$$: validation dataset, not part of training, used to verify the loss
- $$Z_{2test}$$: test dataset. Not part of training or validation, used ensure the model does not overfit.

### Examples

#### Linear lest square regression

1. **Base function**

In model space $$F = \\{ f_{\theta} : x \in \mathbb{R}^{in} \rightarrow \theta^{T}x \in \mathbb{R}^{out} | \theta \in \mathbb{R}^{in \times out} \\}$$, with $$\theta^{T}$$ the transposition of $$\theta$$, linear least square is a loss function that selects the parameter $$\hat \theta$$ that minimises:

$$L(\theta, D) = \frac{1}{N} \sum_{i=1}^{N} || f_{\theta}(x_{i}) - y_{i} ||^{2}$$

(In other words, $$L$$ is the average of the squares of the distance between actual result and expected result)

*Note*: $$\theta^{T}$$ is the transformation of $$\theta$$ and $$\theta^{T}x is $$x \times \theta^{T}$$.

2. **Using matrices**

If we group all inputs and outputs into matrices:
- $$X = [x_{1} | ... | x_{N}]^{T} \in \mathbb{R}^{in \times N}$$,
- $$Y = [y_{1} | ... | y{N}]^{T} \in \mathbb{R}^{N \times out}$$, 
- in other words, $$X$$ and $$Y$$ are transposition of matrices containing all of the inputs (for $$X$$) and outputs ($$Y$$)

$$L$$ can then be written as:

$$L(\theta, D) = \frac{1}{N} ||\theta^{T}X - Y||^{2}$$

(In other words, square of distance between the actual output matrix ($$\theta^{T}X$$) and the expected output matrix $$Y$$, divided by $$N$$, the number of samples in the dataset)
