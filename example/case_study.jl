using JointSurvivalModels
using Turing, Distributions, StatsPlots, CSV, DataFrames, Survival
using ReverseDiff
using LogExpFunctions: logit
using Survival
1
# --------------------------------------------------------------
# data processing
# --------------------------------------------------------------


file_path = joinpath(@__DIR__, "Simulated_Dataset.txt")
file = CSV.File(file_path)
df = DataFrame(file)
# longitudinal data
longit_id = df.ID
longit_time = df.Time
longit_sld = df.SLD
# survival data
surv_id = unique(df.ID)
row_first_entry = [findfirst(x -> x == n, df.ID) for n in surv_id]
surv_time = df[row_first_entry, :T]
surv_indicator = df[row_first_entry, :delta]
first(df,20)

# --------------------------------------------------------------
# specify longitudinal and baseline survival model
# --------------------------------------------------------------

function sld(t, Ψ, tx = 0.0)
    BSLD, g, d, φ = Ψ
    Δt = t - tx
    if t < tx
        return BSLD * exp(g * t)
    else
        return BSLD * exp(g * tx) * (φ * exp(-d * Δt) + (1 - φ) * exp(g * Δt))
    end
end

h_0(t, λ) = 1/λ


# --------------------------------------------------------------
# bayesian model using Turing.jl @model
# --------------------------------------------------------------

@model function identity_link(
    longit_ids,
    longit_times,
    longit_measurements,
    surv_ids,
    surv_times,
    surv_event,
)
    # treatment at study star
    tx = 0.0
    # number of longitudinal and survival measurements
    n = length(surv_ids)
    m = length(longit_ids)

    # ---------------- Priors -----------------
    ## priors longitudinal
    # μ priors, population parameters
    μ_BSLD ~ LogNormal(3.5, 1)
    μ_d ~ Beta(1, 100)
    μ_g ~ Beta(1, 100)
    μ_φ ~ Beta(2, 4)
    μ = (BSLD = μ_BSLD, d = μ_d, g = μ_g, φ = μ_φ)
    # ω priors, mixed/individual effects
    ω_BSLD ~ LogNormal(0, 1)
    ω_d ~ LogNormal(0, 1)
    ω_g ~ LogNormal(0, 1)
    ω_φ ~ LogNormal(0, 1)
    Ω = (BSLD = ω_BSLD^2, d = ω_d^2, g = ω_g^2, φ = ω_φ^2)
    # multiplicative error
    σ ~ LogNormal(0, 1)
    ## prior survival
    λ ~ LogNormal(5, 1)

    ## prior joint model
    γ ~ truncated(Normal(0, 0.5), -0.1, 0.1)

    ## η describing the mixed effects of the population
    η_BSLD ~ filldist(Normal(0, Ω.BSLD), n)
    η_d ~ filldist(Normal(0, Ω.d), n)
    η_g ~ filldist(Normal(0, Ω.g), n)
    η_φ ~ filldist(Normal(0, Ω.φ), n)
    η = [(BSLD = η_BSLD[i], d = η_d[i], g = η_g[i], φ = η_φ[i]) for i = 1:n]
    # Transforms
    Ψ = [  [μ_BSLD * exp(η_BSLD[i]),
            μ_g * exp(η_g[i]),
            μ_d * exp(η_d[i]),
            inverse(logit)(logit(μ_φ) + (η_φ[i])),] 
         for i = 1:n]

    # ---------------- Likelihoods -----------------
    # add the likelihood of the longitudinal process
    for ij in 1:m
        id = Int(longit_ids[ij])
        meas_time = longit_times[ij]
        sld_prediction = sld(meas_time, Ψ[id], tx)
        if isnan(sld_prediction) || sld_prediction < 0
            sld_prediction = 0
        end
        longit_measurements[ij] ~ Normal(sld_prediction, sld_prediction * σ)
    end
    # add the likelihood of the survival model with link
    baseline_hazard(t) = h_0(t, λ)
    for i = 1:n
        id = Int(surv_ids[i])
        id_link(t) = sld(t, Ψ[id], tx)
        censoring = Bool(surv_event[id]) ? Inf : surv_times[id]
        # here we use the JointSurvivalModel
        try
            surv_times[i] ~ censored(JointSurvivalModel(baseline_hazard, β, id_link), upper = censoring)
        catch
        end
    end
end


# --------------------------------------------------------------
# sample posterior with a prior
# --------------------------------------------------------------
# for faster convergence use reverse diffrerentation
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

n = 100
init_params = (
    μ_BSLD=60,μ_d=0.0055, μ_g=0.0015, μ_ϕ=0.2,
    ω_BSLD=0.7 ,ω_d=1.0, ω_g=1.0 , ω_ϕ=1.5, σ=0.18,
    λ=1450, β = 0,
    η_BSLD=zeros(n) ,η_d=zeros(n), η_g=zeros(n) , η_ϕ=zeros(n),
    )


identity_link_model = identity_link(longit_id, longit_time, longit_sld, surv_id, surv_time, surv_indicator)
identity_link_chain = sample(identity_link_model, NUTS(50, 0.9, max_depth = 8), 100, init_params=init_params)
# more samples
# identity_link_chain = sample(identity_link_model, NUTS(1000, 0.9, max_depth = 8), 2000, init_params=init_params)

# --------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------

# posterior
param_names = [:μ_BSLD, :μ_d, :μ_g, :μ_φ, :σ, :κ, :λ, :β]
display(identity_link_chain[param_names])
plot(identity_link_chain[param_names])


# individual predictions
r = range(0,700,2000)
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
km_fit = fit(KaplanMeier, surv_time, BitVector(surv_indicator))
plot!(p_survival_all, r, km_fit.(r), label="Kaplan-Meier")


ylims!(0,1)
plot!(p_survival_all)



# individual predictions based on posterior distributions
Random.seed!(123)
individual = 25
obs_for_individual = longit_id .== individual
individual_obs = longit_sld[obs_for_individual]
individual_time = longit_time[obs_for_individual]
last_obs = individual_time[end]
# timerange to sample
timespan = (0, 1600)
timesteps = 15

r = range(timespan..., timesteps)
sld_timepoints = sort(push!(collect(r),last_obs))

m = length(sld_timepoints)
sld_pred =  Vector{Union{Missing, Float64}}(undef, m)
sld_id = fill(individual, m)
# survival pred
n = 100
s_time = Vector{Union{Missing, Float64}}(undef, n)
s_id = fill(individual, n)
s_event = ones(n)

pred_model = identity_link(sld_id, sld_timepoints, sld_pred, s_id, s_time, s_event)

# Predict using zero $\sigma$

prediction_chn = predict(pred_model, identity_link_chain)
#Plotting longitudinal PPC

chn_longit = group(prediction_chn, :longit_measurements)
longit_quantiles = quantile(chn_longit)
quantiles_df = DataFrame(longit_quantiles)

lower_q = quantiles_df[!,2]
median_q = quantiles_df[!,4]
upper_q = quantiles_df[!,6]
lower_ribbon = median_q - lower_q
upper_ribbon = upper_q - median_q

plot(sld_timepoints, quantiles_df[!, 4],
    ribbon = (lower_ribbon, upper_ribbon),
    title = "Individual $individual",
    label = "SLD model",
    xlabel = "Time",
    ylabel = "SLD",
    legend = :topleft,
)

scatter!(individual_time, individual_obs, label = "Observations")
vline!([last_obs], color = :black, linewidth = 1, label = "last measurement")

# The line "SLD model" represents the median of the posterior predictions and the ribbon represents the 95% quantile.
# Now we can add a conditional survival prediction into this graph starting at the last observation.


#Functionality to work with non-parametric KaplanMeier estimators
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


chn_survival = group(prediction_chn, :surv_times)
indicators = ones(100)
surv_df = select!(DataFrame(chn_survival), Not(1:2))

conditioned_measurements = [row[row .>= last_obs] for row in eachrow(Matrix(surv_df))]
KM = [fit(KaplanMeier, row, indicators[1:length(row)]) for row in conditioned_measurements]
p = [0.025, 0.5, 0.975]
Q = [quantile([npe(km, time_point) for km in KM],p) for time_point in sld_timepoints]
Q_mat = reduce(hcat,Q)
surv_median = Q_mat[2,:]
surv_rib = (surv_median - Q_mat[1,:], Q_mat[3,:] - surv_median)

plot!(twinx(), sld_timepoints, surv_median,
    label = "Joint Survival",
    legend = :topright,
    color = :green,
    ribbon = surv_rib,
    ylims = [0,1]
)
# The line "Overall Survival" represents the median posterior overall survival conditioned to survival until the last observation and the band represent the 95% quantile. This has been calculated non parametrically with generating 100 events for all posterior samples then dropping all event prior to the last overservation. For each posterior sample a Kaplan Meier was calculated and the median and 95% quantile come from their overall survival.
# There is quite some work needed to generate this graph since we combine both the posterior predictive of the longitudinal and survival process. I want to highlight that since we are only using the posterior predictive this plot can be generated for arbitrary joint model formulation. Changing the baseline hazard, link function or longitudinal model does not change anything in the generation of the plot itself.
