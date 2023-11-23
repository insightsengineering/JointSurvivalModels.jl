
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
```@example survival
p1 = plot(LogNormal(0.5), label="κ")
xlims!(p1, (0,8))
p2 = plot(truncated(Normal(8),2,10), label= "λ")
plot(p1, p2, plot_title = "Visualizing the priors")
```

### Model agnostic diagnostics
From the posterior samples we can compute different statistics and diagnostics that help us judge the convergence of the posterior and the descriptiveness of the posterior.

First we use the package `MCMCChains.jl` [docs](https://turinglang.org/MCMCChains.jl/dev/statsplots/) from the Turing framework (no need to load it but `StatsPlots` is needed for plotting). This package is usefull for working with the posterior in the Turing native `Chains` format, without casting it to a DataFrame (which we will do later on for convinience).

Statistics about the posterior samples
```@example survival
summarize(chn_weibull)
```
Trace and posterior densities plots
```@example survival
plot(chn_weibull)
```
Mean development
```@example survival
meanplot(chn_weibull)
```

Autocorelation
```@example survival
autocorplot(chn_weibull)
```

Corner plot
```@example survival
corner(chn_weibull)
```

Another type of statistics we can compute from a posterior is the LOO-CV with Pareto smoothed importance sampling with the package `ParetoSmooth.jl` [docs](https://turinglang.org/ParetoSmooth.jl/stable/) (only docstrings available). We compute this statistics by giving the model instanciation used for finding the posterior `weibull_model` as well as the posterior `chn_weibull`.

```@example survival
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



```@example survival
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

