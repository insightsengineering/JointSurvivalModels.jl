@doc raw"""
    GeneralJointModel <: HazardBasedDistribution <: ContinuousUnivariateDistribution

`GeneralJointModel` is based on the hazard formulaiton:

``h_i(t) = h_0(t) \exp\left(b' \cdot L(M_i (t)) + \beta' \cdot x \right),``

where ``h_0: \mathbb{R} \to \mathbb{R}`` is the baseline hazard function. The term ``L(M(t)): \mathbb{R} \to \mathbb{R}^k, k\in\mathbb{N}`` represents
the link to the longitudinal model(s)  and ``b\in\mathbb{R}^k`` are the link coefficients. Lastly ``x \in\mathbb{R}^l, l\in\mathbb{N}`` the covaraites with coefficients ``\beta\in\mathbb{R}^l``.

# Fields:
- `h₀::Function`: a function in time representing the baseline hazard
- `b::Vector{Real}`: coefficients for links to longitudinal models, should be in the same dimension as `link_m`
- `link_m::Vector{Function}`: unary functions (one argument) in time representing the link to a single or multiple longitudinal models
- `β::Vector{Real}`: coefficients for covariates, should be in the same dimension as `x`
- `x::Vector{Real}`: covariates


# Example
There are constructos for calling `GeneralJointModel` without covariates and for single longitudinal models without the need for arrays
```
using JointModels

GeneralJointModel(identity, 0.01, identity, 0.02, 3)
# corresponds to hazard: identity(t) * exp(0.01 * identity(t) +  0.02 * 3) 
GeneralJointModel(identity, 0.01, identity)
# corresponds to hazard: identity(t) * exp(0.01 * identity(t))
GeneralJointModel(identity,
                    [0.01,-0.02,0.03],
                    [x -> sqrt(x), x -> sin(x)+1, x -> cos(x)^2],
                    [2, 0.3],
                    [ 0, sqrt(2)])
# corresponds to hazard: identity(t) * exp(0.01 * sqrt(t) - 0.02 * (sin(t)+1) + 0.03 * cos(t)^2  + 2 * 0 + 0.3 * sqrt(2))
```
"""
struct GeneralJointModel<:HazardBasedDistribution
    h₀::Function
    b::Vector
    link_m::Vector{Function}
    β::Vector
    x::Vector
    function GeneralJointModel(h₀, b, link_m, β, x)
        # Usability: this allows using singular variables and function instead of arrays
        if !(typeof(link_m) <: Vector) && !(typeof(b) <: Vector)
            link_m = [link_m]
            b = [b]
        end
        if !(typeof(x) <: Vector) && !(typeof(β) <: Vector)
            x = [x]
            β = [β]
        end
        new(h₀, b, link_m, β, x)
    end
end

# Constructor that allows to ommit covariates
GeneralJointModel(h₀, b, link_m) = GeneralJointModel(h₀, b, link_m, [0], [0])


@doc raw"""
The `hazard` for `GeneralJointModel` calculates the hazard according to the formulation
``h(t) = h_0(t) \exp\left(b' \cdot L(M(t)) + \beta' \cdot x \right)``
described in the documentation of `GeneralJointModel`
"""
function hazard(jm::GeneralJointModel, t::Real)
    b, β, x, h₀ = jm.b, jm.β, jm.x, jm.h₀
    link_m_val = [link_m_i(t) for link_m_i in jm.link_m]
    return h₀(t) * exp(β' * x + b' * link_m_val)
end