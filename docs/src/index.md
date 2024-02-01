# JointModels.jl

Documentation for JointModels.jl

This package implements joint models in julia. It was designed to support the modeling of joint models with probabilistic programming, for example using the Turing.jl framework.

To install:
```julia
using Pkg
# needs public listing
#Pkg.add("JointModels")
```


The GeneralJointModel implements a canonical formulation of a joint models. It based on a joint hazard function $h(t) = h_0(t) \exp(b' \cdot L(M(t)))$ with a baseline hazard $h_0$ and a link to joint models $L(M(t))$. For a more detailed explanation see [General Joint Models](@ref GeneralJointModel).

A simple example: The hazard of the exponential distribution $Exp(\alpha)$ is the constant function $x\mapsto \alpha$. For the joint longitudinal model we use a simple cosinus function. The joint hazard is then $h(t) = \alpha \exp(b * cos(t)).

Example:
```julia
constant_alpha(x) = 2
b = 0.01
jm = GeneralJointModel(constant_alpha, b, cos)
```

For the numeric calculation for the distribution a default support (0,10'000) is assumed. In particular the first events happen after $0$ and the interval (0,10'000) should contain nearly all of the probability mass of the target distribution. If you have different starting times for events or a time horizon that exceeds 10'000 then you can manually adjust the support, see [support in Hazard Based Distribution](@ref HazardBasedDistribution)
