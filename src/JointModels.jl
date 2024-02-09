module JointModels

using Distributions
using Integrals: IntegralProblem, solve, QuadGKJL
using DifferentialEquations: ODEProblem, solve, Tsit5
using Roots: find_zero

# stdlib
using Random

export 
    HazardBasedDistribution,
    support,
    hazard,
    cumulative_hazard,
    pdf,
    logpdf,
    cdf,
    ccdf,
    rand,
    JointModel


include("hazarddistribution.jl")
include("jointmodel.jl")

end # module JointModels
