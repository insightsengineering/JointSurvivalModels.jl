# JointModels.jl

Documentation for JointModels.jl

This package implements joint models in julia. It was desinged to support the modeling of joint models with probabelistic programming, for example using the Turing.jl framework.

To install:
```julia
using Pkg
# needs public listing
#Pkg.add("JointModels")
```


The GeneralJointModel implements a canonical formulation of a joint models. It based on a hazard function and is easy to use. See ... link ... for an example.



The HazardBasedDistribution type is the underlying logic behind calculating the log likelihood and generating samples from a distribution based only on the hazard function. In other words only by defining a hazard function HazardBasedDistribution can calculate the 

This package implements a distribution `HazardBasedDistribution` that allows you to specify the hazard in any way you want, including Joint Model formulations.

```@contents
```
