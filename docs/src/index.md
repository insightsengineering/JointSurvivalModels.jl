# JointSurvivalModels.jl

This package implements joint models in Julia. It was designed to support the modeling of joint models with probabilistic programming, for example using the [Turing.jl](https://github.com/TuringLang/Turing.jl) framework. Install it with:

```julia
using Pkg
Pkg.add("JointSurvivalModels")
```

The `JointSurvivalModel` type implements a canonical formulation of joint models. It based on a joint hazard function $h(t) = h_0(t) \exp(\gamma' \cdot L(M(t)))$ with a baseline hazard $h_0$ and a link to joint models $L(M(t))$ and coefficients for the links $\gamma$. For a more detailed explanation see [`JointSurvivalModels`](@ref JointSurvivalModel).



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

For a more instructive example take a look at the documentation [First Example](@ref) or the case study found in `example/` in the project folder.



### Support

For the numeric calculation for the distribution a default support (0.001,10'000) is set. In particular the first events happen after $0$ and the interval (0,10'000) should contain nearly all of the probability mass of the target distribution. If you have different starting times for events or a time horizon that is lower or higher than 10'000 then you should manually adjust the support, see [support in `HazardBasedDistribution`](@ref HazardBasedDistribution). For example:

```julia
JointSurvivalModels.support(dist::HazardBasedDistribution) = (-100, 100)
```
