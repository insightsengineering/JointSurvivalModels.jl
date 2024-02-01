# JointModels.jl

This package implements joint models in Julia. It was designed to support the modeling of joint models with probabilistic programming, for example using the [Turing.jl](https://github.com/TuringLang/Turing.jl) framework. Install it with:

```julia
using Pkg
# needs public listing
#Pkg.add("JointModels")
```

The `GeneralJointModel` implements a canonical formulation of a joint models. It based on a joint hazard function $h(t) = h_0(t) \exp(\gamma' \cdot L(M(t)))$ with a baseline hazard $h_0$ and a link to joint models $L(M(t))$. For a more detailed explanation see the [documentation](https://insightsengineering.github.io/JointModels.jl/dev/).



## Example

The hazard of the exponential distribution $\text{Exp}(\alpha)$ is the constant function $x\mapsto \alpha$. For the joint longitudinal model we use a simple cosinus function. The joint hazard is then $h(t) = \alpha \exp(\gamma * \cos(t))$.

```julia
constant_alpha(x) = 2
γ = 0.01
jm = GeneralJointModel(constant_alpha, γ, cos)
```
Plotting the survival function vs the baseline hazard:
```julia
using StatsPlot
r = range(0,2,100)
plot(r, ccdf(Explonential(1/2), r), label="Baseline survival")
plot!(r, ccdf(jm, r), label="Joint Survival")
```

For a more instructive example take a look at the documentation [first example](https://insightsengineering.github.io/JointModels.jl/dev/FirstExample/) or the case study found in `example/`.

## Contribute

Contributions are welcome, the issue tracker is a good place to start.

## License
This project is licensed under the terms of the MIT license.

