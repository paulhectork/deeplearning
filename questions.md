# Questions

## Chapter 2

### Not sure I understand this equation for MLPs ? 

$$F = \\{ f_{K} \circ ... \circ f_{1} | f_{1} \in F_{1}, ..., F_{K} \\}$$

Là on dirait une diagonale, je verrais plus un produit cartésien.

I would see more that if 
- $$K = 3$$ (there are 3 families of simple parametric functions)
- $$|\theta_k| = 2$$ (there are 2 possible parameter sets for each $$k \in \\{1,...,K\\}$$) 

$$F$$ would be: 

$$\\{ f_{\theta_1k_3} \circ f_{\theta_2k_3} \circ f_{\theta_1k_2} \circ ... \circ f_{\theta_2k1} \\}$$

=> produit cartésien de 
- toutes les fonctions $$f$$ dans chaque famille $$F_k$$
- tous les paramètres $$\theta$$ dans chaque famille $$\Theta_k$$

EN BREF: je comprends pas trop où passent les $$\theta$$ dans cette équation.

### Hidden layers

Why, in the 2-layer MLP, is the 1st layer a hidden layer ? For me, in a 2-layer model, 
- layer 1 is an input layer
- layer 2 is an output layer

And there can be hidden layers only when we have 3+ layers (layer 1 = input, layer 3 = output, layer 2 = hidden).

See: https://en.wikipedia.org/wiki/Hidden_layer
