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
    for i in 1:20
        @test ccdf(num_weibull, i*100) ≈ ccdf(dist_weibull, i*100)
        @test logpdf(num_weibull, i*100) ≈ logpdf(dist_weibull, i*100)
        @test pdf(num_weibull, i*100) ≈ pdf(dist_weibull, i*100)
    end
    # compare generated samples 
    sample_num = rand(num_weibull, 10_000)
    KS_test = ExactOneSampleKSTest(sample_num,dist_weibull)
    @test KS_test.δ < 0.1


end
