# First Examples

## Joint Models
The struct `GeneralJointModel` allows you to implement joint models. Let us consider a simple example. First we describe a model and generate data. We define a nonlinear mixed-effects longitudinal model

```math
m_i(t) = t^(a_i) * (1+cos(b * t)^2),
```
where ``a_i`` is a mixed effects parameter for each individual and ``b`` a population parameter. Next a survival model with baseline hazard

```math
h_0(t) = \alpha/\theta ( t / \theta)^(\alpha -1)
```

with the two parameters ``\alpha`` and ``\theta``. From these we can build a joint model using the identity function ``id: x \mapsto x`` and link coefficient ``c``.

```math
h_i(t) = h_0(t) \exp(c * id(m_i(t))) = \alpha/\theta ( t / \theta)^(\alpha -1) * exp(c * t^(a_i) * cos(b * t)^2).
```

In code:
```julia
parametric_m_i(t, i, a, b) = t^(a[i]) * (1+cos(b * t)^2)
parametric_h_0(t, α, θ) = α/θ *(t/θ)^(1-α)
parametric_joint_model(i, a, b, c, α, θ) = GeneralJointModel(t -> parametric_h_0(t, α, θ), c, t -> m_i(t, i, a, b))
```


To simulate data for 100 individuals we assume ``a_i \overset{~}{\text{iid}} Beta(2,10)`` and ``b = 3, c = 0.02, \alpha = 1.2, \theta = 50``:
```julia
using Distributions
using JointModels
using Random
Random.seed!(222)
n = 100 # number of individuals
a = rand(Product(fill(Beta(10,2), n)))
b = 0.2
c = 0.05
α = 0.6
θ = 70


m(i) = t -> parametric_m_i(t, i, a, b)
h_0(t) = parametric_h_0(t, α, θ)
joint_models = [GeneralJointModel(h_0, c, m(i)) for i in 1:n] # joint models for all individuals
```

Inspecting individual ``1`` and ``2``.


```julia
using StatsPlots
jm = joint_models[1]

r = range(0,50,100)

lm = plot(r, m(1), title="Longitudinal process", label = "Individual 1", color = :blue)
plot!(lm, r, m(2), label = "Individual 2", color = :green)
```
```julia
sm = plot(r, t-> ccdf(joint_models[1], t), label = "Survival individual 1", title="Joint survival process", color = :blue)
plot!(sm, r, t-> ccdf(joint_models[2], t), label = "Survival individual 2", color = :green)
plot!(sm, r, t->ccdf(Weibull(1.2,100),t), label = "Baseline survival", color = :black)
```

To simulate longitudinal measurements ``y_{ij}`` for individual ``i`` at time ``t_ij`` we assume a multiplicative error ``y_{ij} ~ N(m_i(t_{ij}), \sigma * m_i(t_{ij}) ), \sigma = 0.1``. We first simulate ``9`` longitudinal measurements and survival times
```julia
σ = 0.15
t_m = range(1,50,9)
Y = [[rand(Normal(m(i)(t_m[j]), σ * m(i)(t_m[j]))) for j in 1:9] for i in 1:n]
T = [rand(jm) for jm in joint_models]
```
Additionaly we assume right-censoring at ``50`` and no measurements after an event:
```julia
Δ = T .< 50
T = min.(T, 50)
indices = [findlast(T[i] .>= t_m) for i in 1:n]
Y = [Y[i][1:indices[i]] for i in 1:n]
scatter!(lm, t_m[1:indices[1]], Y[1], label="obs individual 1", color = :blue)
scatter!(lm, t_m[1:indices[2]], Y[2], label="obs individual 2", color = :green)
vline!(lm, [T[1]], label="Event time ind 1", color = :blue)
vline!(lm, [T[2]], label="Event time ind 2", color = :green)
```


## Modeling in Turing
One application of this module is the application in bayesian inference frameworks for example `Turing.jl`. For this we choose a suitable prior distribution for the parameters and use the framework to sample the posterior. This works based on the numerical estimation of the log probability density function.

```julia
using Turing
# for better performance with multidimensional distributions (mixed effects)
using ReverseDiff
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

@model function example_longitudinal(Y, t_m, T, Δ)
    n = length(Y)
    
    # longitudianl coef
    beta_1 = 10
    beta_2 = 2
    a ~ filldist(Beta(beta_1,beta_2), n)
    b ~ Uniform(0.1, 0.3)
    # multiplicative error
    σ ~ Exponential(0.2)
    # model
    m(i) = t -> parametric_m_i(t, i, a, b)
    # longitudinal likelihood
    for i in 1:n
        n_i = length(Y[i])
        for j in 1:n_i
            #println([n_i,i,j,a,b, Y[i][j]])
            Y[i][j] ~ Normal(m(i)(t_m[Int(j)]), σ * m(i)(t_m[Int(j)]))
        end
    end
end

longitudinal_chn = sample(example_longitudinal(Y, t_m, T, Δ), NUTS(), 200)
posterior_means = summarize(longitudinal_chn)[:,2]
a_hat = posterior_means[1:n]
b_hat = posterior_means[101]

@model function example_survival(Y, t_m, T, Δ, a, b)
    n = length(Y)
    
    # survival coef
    α ~ Uniform(0.4,1.2) #0.6
    θ ~ LogNormal(4,0.2) #70
    # joint model coef
    c ~ Normal(0,0.03)
    # models
    m(i) = t -> parametric_m_i(t, i, a, b)
    h_0(t) = parametric_h_0(t, α, θ)
    joint_models = [GeneralJointModel(h_0, c, m(i)) for i in 1:n] # joint models for all

    # survival likelihood
    for i in 1:n
        T[i] ~ censored(joint_models[i], upper = 50 + Δ[i]) # if cencored at time 50 then uppder = 50
    end
end
using ForwardDiff
Turing.setadbackend(:forwarddiff)
# using previously sampled posterior for longitudinal process
survival_chn = sample(example_survival(Y, t_m, T, Δ, a_hat, b_hat), NUTS(), 100)


example_model = example(Y,t_m, T, Δ)

# backend for multidimensional distributions
using ReverseDiff
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

init_params = (
    a = fill(0.8, n),
    b = 0.2,
    c = 0.0,
    σ = 0.15
)

example_chn = sample(example_model, NUTS(), 20)#, init_params = init_params)

print(1)

```