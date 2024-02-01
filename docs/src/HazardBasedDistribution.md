# Hazard Based Distributions

Abstract type describing a distribution based on the hazard function.

```@docs
JointModels.HazardBasedDistribution
```



The functionalities implemented for `HazardBasedDistribution`.

```@docs
JointModels.support(dist::HazardBasedDistribution)
JointModels.hazard(dist::HazardBasedDistribution, t::Real)
JointModels.cumulative_hazard(dist::HazardBasedDistribution, t::Real)
Base.rand(rng::AbstractRNG, dist::HazardBasedDistribution)
Distributions.ccdf(dist::HazardBasedDistribution, t::Real)
Distributions.logpdf(dist::HazardBasedDistribution, t::Real)
Distributions.pdf(dist::HazardBasedDistribution, t::Real)
```