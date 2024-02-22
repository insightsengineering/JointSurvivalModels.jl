module JointSurvivalModels

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
    JointSurvivalModel


include("hazarddistribution.jl")
include("jointsurvivalmodel.jl")

end # module JointSurvivalModels
