@docs raw"""
    GeneralJointModel <: HazardBasedDistribution <: ContinuousUnivariateDistribution

`GeneralJointModel` is based on the hazard formulaiton
``h_i(t) = h_0(t) \exp\left( \beta' \cdot x + b' \cdot L(M_i(t)),``
where ``h_0: \mathbb{R} \to \mathbb{R}`` is the baseline hazard, ``x \in\mathbb{R}^l, l\in\mathb{N}`` the covaraites with coefficients ``\beta\in\mathbb{R}^l``. The term ``L(M_i(t)): \mathbb{R} \to \mathbb{R}^k, k\in\mathbb{N}`` represents
the link to the longitudinal model(s) and ``b\in\mathbb{R}^k`` are the link coefficients.

Fields:
    - `h₀`: a function in time representing the baseline hazard
    - `link_m`: function or array of function representing the link to a single or multiple longitudinal models
    - `b`: coefficient(s) for `link_m`, should be in the same dimension as `link_m`
    *- `x`: covariate or vector of covariates
    *- `β`: coefficient(s) for `x`, should be in the same dimension as `x`

* you can use this struct without specifying `x` AND `β` then they will be set to `0` by a constructor.


Usage: ... TO DO ...

"""
struct GeneralJointModel<:HazardBasedDistribution
    h₀
    link_m
    b
    x
    β
end

"""
Constructor if you do not want to consider covariates.
"""
GeneralJointModel(h₀, link_m, b) = GeneralJointModel(h₀, link_m, b, 0, 0)


@doc raw"""
The `hazard` for `GeneralJointModel` calculates the hazard according to the formulation
    ``h_i(t) = h_0(t) \exp\left( \beta' \cdot x + b' \cdot L(M_i(t))``
described in the documentation of `GeneralJointModel`
"""
function hazard(jm::GeneralJointModel, t::Real)
    b, β, x, h₀ = jm.b, jm.β, jm.x, jm.h₀
    link_m_val = [link_m_i(t) for link_m_i in jm.link_m]
    return h₀(t) * exp(β' * x + b' * link_m_val)
end