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

A supervised algorithm takes an input space $$X$$ ($$\mathbb{R}^{n}$$) and **makes a prediction** in space $$Y$$ ($$\mathbb{R}^{m}$$). $$\mathbb{R}^{n}$$ and $$\mathbb{R}^{m}$$ are vector spaces with $$n \in \mathbb{R}$$, $$m \in \mathbb{R}$$, $$n > 0$$, $$m > 0$$.

A model is defined in a **model space** $$F = {f_{\theta} : X \arrow Y|\theta \in \Theta}$$
- the model space is a **function space**, that is a family of all functions that transform input space $$X$$ to $$Y$$.
