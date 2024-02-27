# JointSurvivalModels.jl

This package implements joint models in julia. It was designed to support the modeling of joint models with probabilistic programming, for example using the Turing.jl framework. Install it with:
```julia
using Pkg
# needs public listing
#Pkg.add("JointSurvivalModels")
```


The `JointSurvivalModel` type implements a canonical formulation of joint models. It based on a joint hazard function $h(t) = h_0(t) \exp(\gamma' \cdot L(M(t)))$ with a baseline hazard $h_0$ and a link to joint models $L(M(t))$ and coefficients for the links $\gamma$. For a more detailed explanation see [`JointSurvivalModels`](@ref JointSurvivalModel).


### A simple example

The hazard of the exponential distribution $Exp(\alpha)$ is the constant function $x\mapsto \alpha$. For the joint longitudinal model we use a simple cosinus function. The joint hazard is then $h(t) = \alpha \exp(\gamma * \cos(t))$.

```julia
constant_alpha(x) = 2
γ = 0.01
jm = JointSurvivalModel(constant_alpha, γ, cos)
```

### Support

### Support

For the numeric calculation for the distribution a default support (0.001,10'000) is set. In particular the first events happen after $0$ and the interval (0,10'000) should contain nearly all of the probability mass of the target distribution. If you have different starting times for events or a time horizon that is lower or higher than 10'000 then you should manually adjust the support, see [support in `HazardBasedDistribution`](@ref HazardBasedDistribution). For example:

```julia
JointSurvivalModels.support(dist::HazardBasedDistribution) = (-100, 100)
```
