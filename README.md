# JointModels.jl

Documentation for JointModels.jl

This package implements joint models in julia. It was desinged to support the modeling of joint models with probabelistic programming, for example using the [Turing.jl](https://github.com/TuringLang/Turing.jl) framework.

To install:
```julia
using Pkg
# needs public listing
#Pkg.add("JointModels")
```

The `GeneralJointModel` implements a canonical formulation of a joint models. It based on a joint hazard function $h(t) = h_0(t) \exp(b' \cdot L(M(t)))$ with a baseline hazard $h_0$ and a link to joint models $L(M(t))$. For a more detailed explanaition see (TODO: link to documentation).

A simple example: The hazard of the exponential distribution $Exp(\alpha)$ is the constant function $x\mapsto \alpha$. For the joint longitudinal model we use a simple cosinus function. The joint hazard is then $h(t) = \alpha \exp(b * cos(t)).

Example:
```julia
constant_alpha(x) = 2
b = 0.01
jm = GeneralJointModel(constant_alpha, b, cos)
```
Plotting the survival function vs the baseline hazard:
```julia
using StatsPlot
r = range(0,2,100)
plot(r, ccdf(Explonential(1/2), r), label="Baseline survival")
plot!(r, ccdf(jm, r), label="Joint Survival")
```

For a more instructive example take a look at the (TODO:link to first example) or the publicaiton (TODO: link to publication md/pdf)

