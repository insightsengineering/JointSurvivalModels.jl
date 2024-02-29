"""
    HazardBasedDistribution <: ContinuousUnivariateDistribution

`HazardBasedDistribution` is a type that builds a distribution based 
on a hazard function.

To use this type to formulate a distribution yourself: implement a struct and the hazard function.

```julia
struct LogLogistic <: HazardBasedDistribution
    α::Real
    β::Real
end

function JointSurvivalModels.hazard(dist::LogLogistic, t::Real)
    α, β  = dist.α, dist.β
    return ((β / α) * (t / α) ^ (β - 1)) / (1 + (t / α) ^ β)
end
```

`HazardBasedDistribution` implements numeric integration to calculate the 
cumulative hazard and builds a distribution based it.
To generate samples it solves an ODE and applies inverse transform sampling.
"""
abstract type HazardBasedDistribution <: ContinuousUnivariateDistribution end

"""
This function represents the numeric support of the distribution and return a tuple `(float_start::Float, float_stop::Float)`.
The default value is: `(1e-4, 10_000)`. Note that some distributions are not defined at `0` i.e. Weibull with shape parameter less than 1.

The pdf and hazard are 0 before `float_start` and the numeric integration to calculate the integral over the haazrd in `cumulative_hazard` starts at `beginning`.
When sampling with `rand` an ODE is solved over the support.


# Usage
You should adjust the support according to your data
```julia
JointSurvivalModels.support(dist::HazardBasedDistribution) = (-100, 1000)
```
"""
support(dist::HazardBasedDistribution) = (1e-4, 10_000)

"""
represents ``h(t)``
needs to be implemented for any `struct` that subtypes HazardBasedDistribution
"""
function hazard(dist::HazardBasedDistribution, t::Real) 
    error("You need to implement a hazard.")
end

@doc raw"""
calculates ``H(t) = \int_0^t h(u) \; du `` numerically with a 
Gauss-Konrad procedure.
"""
function cumulative_hazard(dist::HazardBasedDistribution, t_end::Real)
    (start, stop) = support(dist)
    # reformulate for Integrals package (sciml)
    integrals_hazard(t, dist) = hazard(dist, t)
    # some distributions might not be defined at the start: add 0.000001
    ∫h = IntegralProblem(integrals_hazard, start ,float(t_end) ,dist)
    H = solve(∫h, QuadGKJL())[1]
    return H
end

@doc raw"""
Calculation of the ccdf / survival function at time ``t`` based on the 
cumulative hazard

``S(t) = \exp(-H(t)) = \exp(-\int h(u) du)``
"""
function Distributions.ccdf(dist::HazardBasedDistribution, t::Real)
    exp(- cumulative_hazard(dist, t))
end

@doc raw"""
Calculation of the log pdf function at time ``t`` based on the cumulative
hazard 
    
``\log (f(t)) = \log(h(t)\cdot S(t)) = \log( h(t)) - H(t)``
"""
function Distributions.logpdf(dist::HazardBasedDistribution, t::Real)
    log(hazard(dist, t)) - cumulative_hazard(dist, t)
end

"""
Calculation of the pdf function at time ``t`` based on the log pdf.
"""
function Distributions.pdf(dist::HazardBasedDistribution, t::Real)
    exp(logpdf(dist, t))
end

@doc raw"""
Generate a random sample ``t \sim \text{dist}`` via inverse transform 
sampling.
"""
function Distributions.rand(rng::AbstractRNG, dist::HazardBasedDistribution)
    # calculate ode solution of $H(t) = \int_0^t h(u) \, du$, starting at a positive value
    H(nLogS, p, t) = hazard(p, t)
    prob = ODEProblem(H, 0.0, support(dist))
    sol = solve(prob, Tsit5(); p=dist)
    S(t) = exp(-sol(t))
    F(t) = 1 - S(t)
    # inverse sampling: find $t₀$ such that $F(t₀) = x₀$ where $x₀ \sim U(0,1)$
    # via the root of $g(t) := F(t) - x₀$
    x₀ = rand()
    g(t) = F(t) - x₀
    return find_zero(g, support(dist)) 
end
