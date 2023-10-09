module JointModels

using Distributions
using Integrals: IntegralProblem, solve, QuadGKJL
using DifferentialEquations: ODEProblem, solve, Tsit5
using Roots: find_zero

# stdlib
using Random

export 
    HazardBasedDistribution,
    hazard,
    cumulative_hazard


include("hazarddistribution.jl")

end # module JointModels
