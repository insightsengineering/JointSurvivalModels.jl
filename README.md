# JointSurvivalModels.jl

This package implements joint models in Julia. It was designed to support the modeling of joint models with probabilistic programming, for example using the [Turing.jl](https://github.com/TuringLang/Turing.jl) framework. Install it with:

```julia
using Pkg
Pkg.add("JointSurvivalModels")
```

The `JointSurvivalModel` implements a canonical formulation of joint models. It based on a hazard function 
$h(t) = h_0(t) \exp(\gamma' \cdot L(M(t)))$ with a baseline hazard $h_0$ and a link to joint models $L(M(t))$. For a more detailed explanation see the [documentation](https://insightsengineering.github.io/JointSurvivalModels.jl/dev/).



## Example

The hazard of the exponential distribution $\text{Exp}(\alpha)$ is the constant function $x\mapsto \alpha$. For the joint longitudinal model we use a simple cosinus function. The joint hazard is then $h(t) = \alpha \exp(\gamma * \cos(t))$.

```julia
using JointSurvivalModels
constant_alpha(x) = 0.2
γ = 0.5
jm = JointSurvivalModel(constant_alpha, γ, cos)
```
Plotting the survival function vs the baseline hazard:
```julia
using StatsPlots, Distributions
r = range(0,12,100)
plot(r, ccdf(Exponential(1/0.2), r), label="Baseline survival")
plot!(r, ccdf(jm, r), label="Joint Survival")
```

For a more instructive example take a look at the documentation [first example](https://insightsengineering.github.io/JointSurvivalModels.jl/dev/FirstExample/) or the case study found in `example/`.

## Contribute

Contributions are welcome, the issue tracker is a good place to start.

## License
This project is licensed under the terms of the MIT license.

