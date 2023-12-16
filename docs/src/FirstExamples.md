# First Examples

## Joint Models
The struct `GeneralJointModel` allows you to implement joint models. Let us consider a simple example. First we describe a model and generate data. We define a nonlinear mixed-effects longitudinal model

```math
m_i(t) = a * t^(b_i) * cos(c_i * t)^2,
```
where ``a`` is a population parameter and ``b_i, c_i`` are mixed effects. We assume distributions

```math
a ~ Exponential(1.5)
b_i ~ Beta(2,5)
c_i ~ Normal(0,1)
```
The observations are measured with a multiplicative error
```math
e ~ Exponential(0.05)
```
Simulation of such longitudinal data in julia for 100 individual and 10 observations.
```julia
m(t, a, b_i, c_i) = a * t^(b_i) * cos(c_i * t)^2
individual_m(i) = t -> m(t, a, b[i], c[i])

n = 100
time_points = range(0,10,10)

a = 0.5
b = rand(Product(fill(Beta(2,5), n)))
c = rand(MvNormal(zeros(n), ones(n)))

e = 0.02
```
Let the baseline hazard for the survival analysis be the hazard of the Weibull distribution
```math
h_0(t) = \alpha/\theta ( t / \theta)^(\alpha -1)
```
and for individual $i$ we have a contributoin from the true longitudinal model $m_i$. As link to this model we take the identity function $id$ and coeficient $d$. This results in hazard
```math
h_i(t) = h_0(t) \exp(d * identity(m_i(t)))
```
To implement this joint distribution we use the `GeneralJointModel`
```julia
alpha = 1.2
lambda = 20
d = 0.05
h_0(t) = alpha/lambda *(t/lambda)^(1-alpha)
joint_models = [GeneralJointModel(h_0, d,individual_m(i)) for i in 1:n]
```
The parametric hazard is
```math
h_i(t) = alpha/lambda *(t/lambda)^(1-alpha) * exp(d * a * t^(b_i) * cos(c_i * t)^2)
```
Now we can pose the question about the likelihood that individual ``1`` has an event a certain time.
```julia
t = 5
pdf(joint_models[1], t)
```
as well as the probability of survival until a certain time
```julia
ccdf(joint_models[1], t)
```
The module `Distributions.jl` provided a convinient tool ([censoring](https://juliastats.org/Distributions.jl/stable/censored/)) to model left and right cencoring.
```
pdf(censored(joint_models[1], upper = 10), 5)  # likelihood of event occuring at 5
pdf(censored(joint_models[1], upper = 10), 10) # probability of survival until 10
```
Lastly we simulate some measurements that we can use later on for fitting a model.
```julia
# survival
event_times = rand.(joint_models)
event_times = min.(event_times, 10) # censoring at 10
event_indicators = event_times .< 10
# longitudinal measurements with multiplicative error
Y = zeros(n, length(time_points))
for i in range(1,n)
    for (j, t) in enumerate(time_points)
        ob_ij = individual_m(i)(t)
        Y[i,j]  = rand(Normal(ob_ij, ob_ij * e))
    end
end
```

## Fitting a model in Turing
One application of this module is in probabelistic inference. We will simulaneously estimate the mixed effects model parameter as well as the survival parameters and link coefficient.

```julia
using Turing

@model function example(Y, time_points,  event_times, event_indicators, censore_time)
    a ~ Uniform(0.2,1)
    b2 ~ Uniform(2,8)
    b ~ filldist(Beta(2,b2), n)
    c2 ~ Uniform(0.5, 1.5)
    c = rand(MvNormal(zeros(n), c2 * ones(n)))
    # longitudinal models
    individual_m(i) = t -> a * t^(b[i]) * cos(c[i] * t)^2
    # error
    e ~ Exponential(0.05)
    # survival
    alpha ~ Uniform(0.8, 1.6)
    lambda = truncated(Normal(25,10), 10, 30)
    h_0(t) = alpha/lambda *(t/lambda)^(1-alpha)
    # link coeficient
    d ~ Uniform(-0.7, 0.7)
    joint_models = [GeneralJointModel(h_0, d, individual_m(i)) for i in eachindex(event_times)]

    # Likelihood calculations
    # longitudinal measurements
    inds, obs = size(Y)
    for i in 1:inds
        for j in 1:obs
            ob_ij = individual_m(i)(time_points(j))
            Y[i,j] ~ Normal(ob_ij, ob_ij * e)
        end
    end
    # event times and indicators
    for i in eachindex(event_times)
        event_times[i] ~ censored(joint_models[i], upper = event_times[i] + event_indicator[i])
    end
end
```