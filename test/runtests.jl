using Test
using JointModels
using Distributions
using HypothesisTests

@testset "Numerical Hazard" begin
    # Take distribution with known hazard and compare their functionality
    # to numerical evaluations of HazardWeibull
    α, θ = 1.2, 400
    dist_weibull = Weibull(α, θ)

    struct HazardWeibull <: HazardBasedDistribution
        α
        θ
    end

    function JointModels.hazard(dist::HazardWeibull, t::Real)
        α, θ  = dist.α, dist.θ
        return α / θ * (t / θ)^(α - 1)
    end

    num_weibull = HazardWeibull(α, θ)

    # compare survival, and (log)pdf
    test_values = 10.0 .^ collect(-2:3)
    for t in test_values
        @test ccdf(num_weibull,t) ≈ ccdf(dist_weibull, t)
        @test logpdf(num_weibull, t) ≈ logpdf(dist_weibull, t)
        @test pdf(num_weibull, t) ≈ pdf(dist_weibull, t)
    end
    # compare generated samples 
    sample_num = rand(num_weibull, 10_000)
    KS_test = ExactOneSampleKSTest(sample_num,dist_weibull)
    @test KS_test.δ < 0.1

end

@testset "Joint Model interface tests" begin
    # jm1 tests constructor for non array arguments
    jm1 = GeneralJointModel(identity, 0.01, identity, 0.02, 3)
    hazard1(t) = identity(t) * exp(0.01 * identity(t) +  0.02 * 3) 
    # jm2 tests constructor with no covariates
    jm2 = GeneralJointModel(identity, 0.01, identity)
    hazard2(t) = identity(t) * exp(0.01 * identity(t))
    # jm3 tests multiple longitudinal models and and covariates
    jm3 = GeneralJointModel(identity,
                            [0.01,-0.02,0.03],
                            [x -> sqrt(x), x -> sin(x)+1, x -> cos(x)^2],
                            [2, 0.3],
                            [ 0, sqrt(2)])
    hazard3(t) = identity(t) * exp(0.01 * sqrt(t) - 0.02 * (sin(t)+1) + 0.03 * cos(t)^2  + 2 * 0 + 0.3 * sqrt(2))

    for t in [0.125, 0.25, 0.5, 1, 2, 4]
        @test hazard(jm1, t) == hazard1(t)
        @test hazard(jm2, t) == hazard2(t)
        # approx is used here since for t = 4 there are computational differences due to complex calculation in exponential
        @test hazard(jm3, t) ≈ hazard3(t)
    end

end

@testset "Joint Model interface tests" begin
    # jm1 tests constructor for non array arguments
    jm1 = GeneralJointModel(identity, 0.01, identity, 0.02, 3)
    hazard1(t) = identity(t) * exp(0.01 * identity(t) +  0.02 * 3) 
    # jm2 tests constructor with no covariates
    jm2 = GeneralJointModel(identity, 0.01, identity)
    hazard2(t) = identity(t) * exp(0.01 * identity(t))
    # jm3 tests multiple longitudinal models and and covariates
    jm3 = GeneralJointModel(identity,
                            [0.01,-0.02,0.03],
                            [x -> sqrt(x), x -> sin(x)+1, x -> cos(x)^2],
                            [2, 0.3],
                            [ 0, sqrt(2)])
    hazard3(t) = identity(t) * exp(0.01 * sqrt(t) - 0.02 * (sin(t)+1) + 0.03 * cos(t)^2  + 2 * 0 + 0.3 * sqrt(2))

    for t in [0.125, 0.25, 0.5, 1, 2, 4]
        @test hazard(jm1, t) == hazard1(t)
        @test hazard(jm2, t) == hazard2(t)
        # approx is used here since for t = 4 there are computational differences due to complex calculation in exponential
        @test hazard(jm3, t) ≈ hazard3(t)
    end    

end
