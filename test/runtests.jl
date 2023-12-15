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
    jm = GeneralJointModel(identity, 0.01, identity, 0.02, 3)

    hazard(jm, 1)

    jm2 = GeneralJointModel(identity, 0.01, identity)

    jm3 = GeneralJointModel(identity,
                            [0.01,0.02,0.03],
                            [x -> sqrt(x), x -> x^0.1, x -> 1 + cos(x)],
                            [0.1, 2, 30],
                            [12.3, 0, sqrt(2)])

    

end
