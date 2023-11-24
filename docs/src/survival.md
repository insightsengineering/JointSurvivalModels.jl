# Survival Analysis using Hazard based formulations and Bayesian sampling
This workflow shows Bayesian sampling of a survival process and introduces hazard based formulations for the survival distribution (This is done with the anticipation of joint models which can have no analytical formulation of the pdf and survival function (ccdf)).

## Basics of the Turing framework

First we load packages
```@example survival
using JointModels
using Distributions, StatsPlots, DataFrames, Turing, Survival
```


Load breastcancer data `bc` from R package `flexsurv` (via link to `.rda` file)
```@example survival
using RData
url = "https://github.com/chjackson/flexsurv-dev/raw/master/data/bc.rda"
temp = download(url)
parsed = RData.load(temp)
bc = parsed["bc"]
```


Plot Kaplan Meier curves for the three groups "Good", "Medium", "Poor" with the `Survival.jl` package [docs](https://juliastats.org/Survival.jl/stable/).



```@example survival
# refere to column name with symbol :column_name
groups = groupby(bc, :group)
r = range(0,8,2000)
p_survival_all = plot(title = "Overall Survival", ylabel = "Survival percentage", xlabel = "Years")
function (km::KaplanMeier)(t::Real)
    time_points = km.events.time
    survival = km.survival
    if t < time_points[1]
        return 1
    else
        id = findlast(x -> x <= t, time_points) 
        return survival[id]
    end
end
for (index, group) in enumerate(groups)
    # fit using the Survival package
    km_fit = fit(KaplanMeier, group.recyrs, BitVector(group.censrec))
    plot!(p_survival_all, r, km_fit.(r), label="KM $index")
end
ylims!(0,1)
plot!(p_survival_all)
```

For simplicity we only continue the group with lable "Good". 

```@example survival
g1 = groups[1]
p_survival = plot(title = "Overall Survival", ylabel = "Survival percentage", xlabel = "Years")
km_fit = fit(KaplanMeier, g1.recyrs, BitVector(g1.censrec))
plot!(p_survival, r, 1km_fit.(r), label="KM")
ylims!(0,1)
plot!(p_survival)
```


Let us build a bayesian model with `Turing.jl` [docs](https://turinglang.org/stable/) and sample its posterior. To describe the censoring process the package `Distributions` has the functionality to censor [docs](https://juliastats.org/Distributions.jl/stable/censored/) notice that this only provides for right and/or left censoring.

```@example survival
@model function weibull_breastcancer(time_obs, indicators)
    # Priors
    κ ~ LogNormal(0.5)
    λ ~ truncated(Normal(8),2,10)
    # Likelihood
    n = size(time_obs, 1)
    for id in 1:n
        if Bool(indicators[id]) # uncensored
            time_obs[id] ~ Weibull(κ,λ)
        else # right(upper) censored
            time_obs[id] ~ censored(Weibull(κ,λ), upper = time_obs[id])
        end
    end
end
# data from first group
surv_times = Vector(g1.recyrs)
event_indicators = Vector(g1.censrec)
# setup Turing model and sample posterior for 500 samples
weibull_model = weibull_breastcancer(surv_times, event_indicators)
chn_weibull = sample(weibull_model, NUTS(), 500)
display(chn_weibull)
```




Plotting the priors. With the package `StatsPlots` we can plot the pdf of a distribution by just calling `plot` on int.
```julia
p1 = plot(LogNormal(0.5), label="κ")
xlims!(p1, (0,8))
p2 = plot(truncated(Normal(8),2,10), label= "λ")
plot(p1, p2, plot_title = "Visualizing the priors")
```

### Model agnostic diagnostics
From the posterior samples we can compute different statistics and diagnostics that help us judge the convergence of the posterior and the descriptiveness of the posterior.

First we use the package `MCMCChains.jl` [docs](https://turinglang.org/MCMCChains.jl/dev/statsplots/) from the Turing framework (no need to load it but `StatsPlots` is needed for plotting). This package is usefull for working with the posterior in the Turing native `Chains` format, without casting it to a DataFrame (which we will do later on for convinience).

Statistics about the posterior samples
```julia
summarize(chn_weibull)
```
Trace and posterior densities plots
```julia
plot(chn_weibull)
```
Mean development
```julia
meanplot(chn_weibull)
```

Autocorelation
```julia
autocorplot(chn_weibull)
```

Corner plot
```julia
corner(chn_weibull)
```

Another type of statistics we can compute from a posterior is the LOO-CV with Pareto smoothed importance sampling with the package `ParetoSmooth.jl` [docs](https://turinglang.org/ParetoSmooth.jl/stable/) (only docstrings available). We compute this statistics by giving the model instanciation used for finding the posterior `weibull_model` as well as the posterior `chn_weibull`.

```julia
#using ParetoSmooth
#psis_loo(weibull_model, chn_weibull)
```

There are also other possibilities to analyze the posterior (which might have feature you desire) using utilities that depend on other languages. One good option would be `ArviZ.jl` [docs](https://julia.arviz.org/ArviZ/stable/) that is a actively maintained interface to a python library by the same name (thus you need to install the python package as well). For such solutions you typically also need to transfer the posterior into the formate the spesific package expects.




We can also compute diagnostics that depend on the model formulation. For example we can add a naive estimator to the Survival plot. I define the naive estimator as the estimator with the mean values of the posterior samples **independently**.

```@example survival
# mean values of posterior
κ_hat, λ_hat = summarize(chn_weibull)[:,2]
# braodcast ccdf/survival over x
S(t) = ccdf(Weibull(κ_hat, λ_hat), t)
plot!(p_survival ,r , S.(r) , label="naive Weibull estimator")
```
Notice that here we take assumptions about the model. We expect it to be a `Weibull` model and that the first two parameters in the posterior represent $\kappa$ and $\lambda$. If we use a different distribution or change around the parameters we can not just run this code and get a sensible output. We would therefore need to change the code!!



### Posterior predictions

The Turing framework has the functionality `predict`. For every sample in the posterior `chn_weibull` we can predict time to event according to the parameters describing the survival distribution of that sample. Note: here "predict a time to event" means to sample a `Weibull` given the posterior samples of $\kappa$ and $\lambda$ of every posterior sample in the MCMChain i.e. drawing $t \sim \text{Weibull}(\kappa', \lambda')$ for all $(\kappa', \lambda')$ in the posterior distribution `chn_weibull`.

We initialize a model with a vector of $229$ (size of `g1`) `missing` values as survival data. Turing will then draw a sample for each missing value. We use $100$ since later on in the case of Join Models the distribution of all individuals will be different.



```julia
# setup model with undef/missing values for event times
# for event indicator we need to give a ones vector since we want to sample from the uncensored distribution
surv_missing =  Vector{Union{Missing, Float64}}(undef, size(g1,1))
pred_weibull_model = weibull_breastcancer(surv_missing, ones(size(g1,1)))
# predict based on the samples in the `chn`
predictions = predict(pred_weibull_model, chn_weibull)
```


## Extending Distributions
The distributions `Weibull` is provided by the `Distributions.jl` package, it provides 69 univariate distributions ([docs](https://juliastats.org/Distributions.jl/stable/univariate/#Index)). Note that for example the Log-Logistic distribution is missing, which might be interesting for Survival Analysis. You can implement your own distribution yourself ([docs](https://juliastats.org/Distributions.jl/stable/extends/#Create-a-Distribution)) and then use it inside the Turing framework ([docs](https://turinglang.org/dev/docs/using-turing/advanced#how-to-define-a-customized-distribution)) for bayesian sampling. For the Turing framework the functionalities `rand` and `logpdf` are needed.

Since we will use hazard based formulations of survival distributions for joint models, I designed a support structure that allows you to define a distribution based on its hazard function.

```julia
#| code-fold: true
#| code-summary: "Code for hackers"
using Integrals, DifferentialEquations, Random, Roots, Distributions


abstract type HazardBasedDistribution <: ContinuousUnivariateDistribution end

@doc raw"""
represents ``h(t)``
needs to be implemented for any `struct` that subtypes HazardBasedDistribution
"""
function hazard(dist::HazardBasedDistribution, t::Real) end

@doc raw"""
Different argument order to comply with sciml (Integration)
"""
sciml_hazard(t::Real, dist::HazardBasedDistribution) = hazard(dist, t)

@doc raw"""
calculates ``H(t) = \int_0^t h(u) \; du `` numerically with a Gauss-Konrad procedure
"""
function cumulative_hazard(dist::HazardBasedDistribution, t::Real)
    # integrate from 0.000001 to $t$ (some hazards are not defined at 0)
    ∫h = IntegralProblem(sciml_hazard ,1e-6 ,t ,dist)
    H = solve(∫h, QuadGKJL())[1]
    return H
end

@doc raw"""
Calculation of the ccdf / survival function at time `t` based on the cumulative hazard ``S(t) = \exp(H(t)) = \exp(\int h(u) du)``
"""
function Distributions.ccdf(dist::HazardBasedDistribution, t::Real)
    exp(- cumulative_hazard(dist, t))
end

@doc raw"""
Calculation of the log pdf function at time `t` based on the cumulative hazard ``\log f(t) = h(t)\cdot S(t) = \log h(t) - H(t)``
"""
function Distributions.logpdf(dist::HazardBasedDistribution, t::Real)
    log(hazard(dist, t)) - cumulative_hazard(dist, t)
end

@doc raw"""
Calculation of the pdf function at time `t` based on the log pdf
"""
function Distributions.pdf(dist::HazardBasedDistribution, t::Real)
    exp(logpdf(dist, t))
end



H(nLogS, p, t) = sciml_hazard(t,p)
@doc raw"""
Generate a random sample ``t \sim \text{dist}``
"""
function Distributions.rand(rng::AbstractRNG, dist::HazardBasedDistribution)
    num_end = 1_000_000
    # calculate ode solution of $H(t) = \int_0^t h(u) \, du$
    prob = ODEProblem(H ,0.0 ,(1e-4,num_end))
    sol = solve(prob, Tsit5(); p=dist)
    # calculate cdf $F(t)$, via survival $S(t)$
    S(t) = exp(-sol(t))
    F(t) = 1 - S(t)
    # inverse sampling: find $t₀$ such that $F(t₀) = x₀$ where $x₀ \sim U(0,1)$
    # via the root of $g(t) := F(t) - x₀$
    x₀ = rand()
    g(t) = F(t) - x₀
    return find_zero(g,(1e-4, num_end)) 
end
```
Here is an implementation of the LogLogistic: (This is a fabricated example, since there is a well known analytical formulation of the pdf, ccdf/survival etc. It would be better to implement this distribution with analytical calculations) We use the formulation found on [Wikipdedia](https://en.wikipedia.org/wiki/Log-logistic_distribution).

```julia
struct LogLogistic <: HazardBasedDistribution
    α::Real
    β::Real
end

function hazard(dist::LogLogistic, t::Real)
    α, β  = dist.α, dist.β
    return ((β/α)*(t/α)^(β-1))/(1 + (t/α)^β)
end
```
To convince ourselves of this implementation we showcase the analytical pdf, numerical pdf and a histogram of 10'000 measurements:

```julia
α, β = 1, 8
n = 10_000
T = range(0, 4, length=200)
# analytical pdf
f_analytical(t) = ((β/α)*(t/α)^(β-1)) / (1 + (t/α)^β)^2
# numerical pdf
dist = LogLogistic(α, β)
f_hazardbased(t) = pdf(dist, t)
# 10_000 samples
measurements = [rand(dist) for i in 1:n]
# plotting
histogram(measurements, bins=90, normalize=:pdf, label = "Normalized histogram, $n measurements")
plot!(T, f_analytical.(T), label="Analytical formulation", linewidth = 3)
plot!(T, f_hazardbased.(T), label="Numerical integration of hazard",linestyle=:dash, linewidth = 3)
```


Now we can use `LogLogistic` in Turing
```julia
@model function loglogistic_breastcancer(time_obs, indicators)
    # Priors
    α ~ truncated(Normal(8,2),1,20)
    β ~ truncated(Normal(2),0.2,4)
    n = size(time_obs, 1)
    for id in 1:n
        if Bool(indicators[id]) # uncensored
            time_obs[id] ~ LogLogistic(α, β)
        else # right censored
            time_obs[id] ~ censored(LogLogistic(α, β), upper = time_obs[id])
        end
    end
end
# setup Turing model and sample posterior
loglogistic_model = loglogistic_breastcancer(surv_times, event_indicators)
chn_loglogistic = sample(loglogistic_model, NUTS(), 500)

```

Posterior of `LogLogistic` model.
```julia
plot(chn_loglogistic)
```

Adding the naive `LogLogistic` estimator.

```julia
x = range(0,8,200)
α_hat, β_hat = summarize(chn_loglogistic)[:,2]
# braodcast ccdf/survival over x
y = ccdf.(LogLogistic(α_hat, β_hat), x)
plot!(p_survival ,x ,y, label="naive LogLogistic estimator")
```



We can also implement the Weibull ourselves
```julia
#| code-fold: true
#| code-summary: "Weibull implementation"
struct HazardWeibull <: HazardBasedDistribution
    κ::Real
    λ::Real
end

function hazard(d::HazardWeibull, t::Real)
    return d.κ / d.λ * (t / d.λ)^(d.κ - 1)
end


@model function hazard_weibull_breastcancer(time_obs, indicators)
    # Priors
    κ ~ truncated(LogNormal(0.5),0.4,2)
    λ ~ truncated(Normal(8),0.1,10)
    n = size(time_obs, 1)
    for id in 1:n
        if Bool(indicators[id]) # uncensored
            time_obs[id] ~ HazardWeibull(κ, λ)
        else # right censored
            time_obs[id] ~ censored(HazardWeibull(κ, λ), upper = time_obs[id])
        end
    end
end
# setup Turing model and sample posterior
hazard_weibull_model = hazard_weibull_breastcancer(surv_times, event_indicators)
chn_hazard_weibull = sample(hazard_weibull_model, NUTS(), 500)
```

Here we can compare the posterior statistics of the hazard Weibull model with the analytical implementation of the Distribution package.

Analytical formulation:
```julia
summarize(chn_weibull)
```
Hazard based numeric formulation:
```julia
summarize(chn_hazard_weibull)
```


Q: How would you extend the structure of `LogLogsitic` or `Weibull` to include covariate data? (and covariate coefficients?)

<details>
<summary>Idea</summary>
We build a new `struct` with a vector for covariates $x$ and a vector for covariate coeficients $b$ and extend the hazard formulation by multiplying with $exp(b_1 \cdot x_1 + \dots + b_n \cdot x_n)$.

```julia
#| eval: false
struct HazardWeibullCovariate <: HazardBasedDistribution
    κ::Real
    λ::Real
    x::Vector{Real}
    b::Vector{Real}
end

function hazard(d::HazardWeibullCovariate, t::Real)
    return d.κ / d.λ * (t / d.λ)^(d.κ - 1) * exp(sum(d.b .* d.x))
end
```

In the Turing `@model` we then also give a prior for the covariate parameters.

This is already a step towards building Joint Models later on!
</details>




## Survival Diagnostics

To create diagnostics for a Bayesian model we might want to calculate some quantities for samples in our posterior distribution of models. For this I developed a `DisgnosticHelper`
that allows you to save quantities based on posterior samples using the Turing framework to parse the posterior.

```julia
#| code-fold: true
#| code-summary: "Disgnostic Helper"
"""
The elegance of this helper is that Turing does all the legwork for parsing the posterior samples and keeping an eye on any changes in the model. (This will be helpfull when we have individual parameters for the mixed effects of the longitudinal model)
If you want to change the model you have to do minimal changes to the diagnostics to still get the same measurements.
"""
struct DiagnosticHelper <: ContinuousUnivariateDistribution
    value::Real
end

function Distributions.rand(rng::AbstractRNG, dist::DiagnosticHelper)
    return dist.value
end

function Distributions.logpdf(dist::DiagnosticHelper, t::Real)
    return 0
end
```

Let us calculate cox-snell residuals i.e. the cumulative hazard $H(t)$ (here we do not consider covariates) for the individuals using the `LogLogsitic` model. We can calculate these residuals in the initial sampling step or use the posterior `chn_loglogistic` we have calculated before. In both cases we have to extend the `@model` definition.



```julia
using DynamicPPL
@model function loglogistic_breastcancer_cox_snell(time_obs, indicators)
    # Priors
    α ~ truncated(Normal(8,2),1,20)
    β ~ truncated(Normal(2),0.2,4)
    n = size(time_obs, 1)
    # this line is all the magic, here we calculate the cumulative hazard for all the individuals and save them into .the chains output
    cox_snell_residual ~ arraydist([DiagnosticHelper(
                                    cumulative_hazard(LogLogistic(α, β), time_obs[id]) # <1>
                                ) for id in 1:n]) # <1>
    """ # <2>
    for id in 1:n # <2>
        if Bool(indicators[id]) # uncensored # <2>
            time_obs[id] ~ LogLogistic(α, β) # <2>
        else # right censored # <2>
            time_obs[id] ~ censored(LogLogistic(α, β), upper =  time_obs[id]) # <2>
        end # <2>
    end # <2>
    """ # <2>
end
# setup Turing model
loglogistic_model_res = loglogistic_breastcancer_cox_snell(surv_times, event_indicators)
loglogistic_cox_snell_residuals_chn = predict(loglogistic_model_res, chn_loglogistic) # <3>
```
1. Calculates $H(t)$ based on hazard of posterior parametric distriubtion for all individuals

2. Here we leave out (comment) the likelihood formulation since this would generate individual prediction of survival time, which we might not be interested in.

3. When predicting the `DiagnosticHelper` will be samples which returns the value we want to generate.




From this output we can plot the cox snell residuals for the posterior distribution against the cumulative hazard of the Kaplan Meier estimator. Per individual we calculate the mode of all posterior cox snell residuals as well as the quantiles. We then plot this versus the cumulative hazard of the Kaplan-Meier estimator.

```julia
function plot_cox_snell(cox_snell_residuals_chn, surv_times, event_indicators; column_regex = r"cox_snell_residual.*")
    cox_snell_df = DataFrame(cox_snell_residuals_chn)
    posterior_residuals = select!(cox_snell_df, column_regex)
    cox_snell_modes = mode.(eachcol(posterior_residuals))

    # quantiles = quantile(cox_snell_residuals_chn)
    # relative_error_bars = (cox_snell_modes .- quantiles[:,2], quantiles[:,6] .- cox_snell_modes)

    # fit kaplan meire
    km_fit = fit(KaplanMeier, surv_times, BitVector(event_indicators))
    # sort
    surv_times_idx = [findfirst(x -> x == t, km_fit.events.time) for t in surv_times]
    estimated_c_hazard_values = [-log.(km_fit.survival[i]) for i in surv_times_idx]

    p_cox_snell = scatter(cox_snell_modes, estimated_c_hazard_values,
        xlabel ="Cox Snell Residuals",
        ylabel = "Kaplan-Meier Cumulative hazard estimate",
        label = false,
        title = "Mode of Cox-Snell residuals for posterior models",
        color = :lightblue,
        # xerror = relative_error_bars,
        markersize = 1
    )
    plot!(p_cox_snell, x -> x, color=:black, linewidth = 2, label = "identity")
end
plot_cox_snell(loglogistic_cox_snell_residuals_chn, surv_times, event_indicators)
```
Now comparing to `Weibull`
```julia
@model function weibull_breastcancer_cox_snell(time_obs, indicators)
    # Priors
    κ ~ truncated(LogNormal(0.5),0.4,2)
    λ ~ truncated(Normal(8),0.1,10)
    n = size(time_obs, 1)
    cox_snell_residual ~ arraydist([DiagnosticHelper(
                                    cumulative_hazard(HazardWeibull(κ, λ), time_obs[id])
                                ) for id in 1:n])
end
weibull_model_res = weibull_breastcancer_cox_snell(surv_times, event_indicators)
weibull_cox_snell_residuals_chn = predict(weibull_model_res, chn_hazard_weibull)
plot_cox_snell(weibull_cox_snell_residuals_chn, surv_times, event_indicators)
```

Suppose we have some covariate for out data (here we just uniformly sample from $(0,1)$ to showcase the plot).
```julia
covariates = rand(size(g1,1))
```
We can also plot the martingale residuals by the formulation $M = \delta - H(t)$ against this covariate, where $H(t)$ represents the cox snell residuals, here just the cumulative hazard.

```julia
using Loess
function plot_martingale(cox_snell_residuals_chn, surv_times, event_indicators, covariates)
    cox_snell_df = DataFrame(cox_snell_residuals_chn)
    posterior_residuals = select!(cox_snell_df, r"cox_snell_residual.*")
    cox_snell_modes = mode.(eachcol(posterior_residuals))

    martingale_residual_modes = event_indicators .- cox_snell_modes

    p_martingale = scatter(covariates, martingale_residual_modes,
        xlabel = "Covariate",
        ylabel = "Martingale residuals",
        label = false,
        title = "Martingale residual modes of posterior",
        color = :blue,
        markersize = 3,
        markerstrokewidth = 0.2
    )

    # loess
    moving_average = loess(covariates, martingale_residual_modes)
    r = range(extrema(covariates)...,200)
    plot!(p_martingale, r,predict(moving_average,r), color = :black, linewidth = 1, label = "loess")
    plot!(p_martingale, x -> 0, color = :black, linewidth = 0.5, label = false)
end
plot_martingale(loglogistic_cox_snell_residuals_chn, surv_times, event_indicators, covariates)

```
And for the hazard Weibull model
```julia
plot_martingale(weibull_cox_snell_residuals_chn, surv_times, event_indicators, covariates)
```

### Posterior predictive

To showcase the posterior predictive of a posterior of a bayesian sampling I want to showcase a non-parametric plot. For this we will generate survival data via the Turing framework and then compute Kaplan-Meier estimators. My idea here is that this plotting is agnostic of the model you build. It just assumes generated survival data of your model. Above we used `predict` to generate samples for the posterior with a Weibull model.

```julia
display(predictions)
```
For each sample in the posterior we generated $100$ samples. We can estimate its survival via a Kaplan Meier estimator. We end up with $500$ kaplan meier estimators. Then we plot a 95% quantile version of all these estimators at the choosen timepoints (`range(0,8,length=200)`).

```julia
function npe(km_fit::KaplanMeier, t::Real)
    time_points = km_fit.events.time
    survival = km_fit.survival
    if t <= time_points[1]
        return 1
    elseif t >= time_points[end]
        return survival[end]
    else
        id = findfirst(x -> x >= t, time_points)
        return survival[id]
    end
end

function ribbon_npe_ppc(predictions_chn, r; column_regex = r"time_obs.*", p=[0.025, 0.5, 0.975])
    chn_df = DataFrame(predictions_chn)
    predictions_df = select!(chn_df, column_regex)
    indicators = ones(size(predictions_df, 2))
    KM = [fit(KaplanMeier, row, indicators) for row in eachrow(predictions_df)]
    Q = [quantile([npe(km, time_point) for km in KM],p) for time_point in r]
    Q_mat = reduce(hcat,Q)
    return (surv = Q_mat[2,:], rib = (Q_mat[1,:], Q_mat[3,:]))
end

r = range(0,8,20)
km_fit = fit(KaplanMeier, surv_times, BitVector(event_indicators))
surv = [npe(km_fit, time_point) for time_point in r]
surv, rib = ribbon_npe_ppc(predictions, r)
ribb = (surv .- rib[1], rib[2] .- surv)

plot!(p_survival,r, surv, ribbon = ribb, label="KM with ppc")
```

And plotting only Kaplan Meier of the observation and the ppc.

```julia
p_survival_ppc = plot(title = "Overall Survival", ylabel = "Survival percentage", xlabel = "Years")
km_fit = fit(KaplanMeier, g1.recyrs, BitVector(g1.censrec))
plot!(p_survival_ppc, km_fit.events.time, km_fit.survival, label="KM of group \"Good\"")
ylims!(0,1)
plot!(p_survival_ppc, r, surv, ribbon = ribb, label="PPC")
plot(p_survival_ppc)
```





