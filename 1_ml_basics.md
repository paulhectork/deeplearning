# ML Basics

## Intro

Contrary to most scientific methods (i.e., physics), ML isn't about building a mathematical model of reality. It is about being able to predict an outcome based on a set of data points, without formalizing a model of reality. 

Different learning scenarios:
- **supervised learning**: in your dataset, you include examples of desired outputs for your model. 
    - => generalisation issues
- **unsupervised learning**: the input is raw data points without desired outputs.
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
- $$x$$ a single set of data points $$x$$.

How do we measure the efficiency of $$f$$ ?

Risk and loss are **formalizations** of the idea of *best prediction*. In short, 
- for a single set of parameters, there is **1 risk, several losses**
- **cost** measures the quality of a single prediciton for a single set of data points
- **risk** is the average of all losses for all data points for this set of parameters.

**$$l(f(x),y)$$ - cost function**: $$l: Y \times Y \rightarrow \mathbb{R}$$
- it can be expressed as: $$l(y',y)$$, with 
    - $$y$$: the expected output
    - $$y'$$: the actual output of a prediction $$f$$
    - in other words, $$l$$ compares actual output of $$f$$ to expected output $$y$$

**$$R$$ - Risk = average loss over all data points**: 
- in **statistical learning**, $$R(f) = \mathbb{E}_{(x,y) \sim Z}[l(f(x),y)]$$. 
    - we average losses for all data points $$(x,y)$$ randomly sampled from a distribution $$Z$$.
    - $$Z$$ is a probability distribution made of tuples $$(x,y)$$ where $$x$$ is the input data points and $$y$$ is the output data points.
    - in practice, the entire $$Z$$ cannot be known (we cannot know the entierty of the dataset a model will ever be used on) => we need another measure
- since $$Z$$ can never be known, we **define $$Z_{2}$$**, a known subset of $$Z$$ such that $$(x,y) \in Z_{2}$$ and $$Z_{2} \in Z$$.
- **empirical risk** is a measure of risk where all values of $$Z_{2}$$ are known:
    - $$R_{Z_{2}}(f) = \frac{1}{|Z_{2}|} \sum{z \in Z_{2}} l(f(x),y)$$
    - $$Z_{2}$$ is a known set of samples $$(x,y) \in Z$$ => $$Z$$ is our dataset and a subset of all possible occurrences $$Z$$. 

**$$L(\theta, Z_{2})$$ - loss function = cost for all data points for a single set of parameters**
- it is expressed as: $$L(\theta, Z_{2})$$



