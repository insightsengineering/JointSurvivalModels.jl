---
title: 'JointSurvivalModels.jl: Numeric approach to joint models'
tags:
  - Survival analysis
  - Nonlinear
  - Biostatistics
  - Time-to-event
  - Mixed-effects model
authors:
  - name: Yannik Ammann
    orcid: 0009-0009-3296-3577
    affiliation: "1, 2"
  - name: Daniel Sabanés Bové
    affiliation: 2
  - name: Francois Mercier
    orcid: 0000-0002-5685-1408
    affiliation: 2
  - name: Douglas O. Kelkhoff 
    orcid: 0009-0003-7845-4061 
    affiliation: 2
  
affiliations:
 - name: ETH Zürich, Switzerland
   index: 1
 - name: Hoffmann-La Roche Ltd.
   index: 2
date: 02 February 2024
bibliography: paper.bib
---

# Summary
`JointModels.jl` implements a numerical approach to define a distribution based on the hazard function. In particular, this provides a mechanism for defining joint models of time-to-event data and longitudinal measurements. Using numerical integration, the likelihood of events can be calculated, allowing Bayesian inference frameworks to sample the posterior distribution of the model's parameters. Additionally, this implementation can generate samples of joint models. This allows its use in simulations and predictions, which are common in Bayesian workflows [@BayesianWorkflow].


# Statement of need

Over the last decade, there has been a growing interest in joint models for longitudinal and time-to-event outcome data. This is the case in clinical research where biomarkers are often measured repeatedly over time, with the expectation that they may provide insights into the likelihood of a long-term clinical outcome such as disease progression or adverse events. As a consequence, joint models have been proposed to leverage this data and be used for the prediction of individual risks, the evaluation of a biomarker for a clinical outcome, and for making inferences about treatment.

In oncology, it is well known that the individual risk of death correlates with the treatment-induced changes in tumor size over time [@Tardivon2019]. Because pharmacologic effects are typically transient, nonlinear mixed-effect models are required to capture central tendency and inter-individual variability in tumor size, while parametric models can be used for analyzing time to death. Joint models have been used in other therapeutic areas such as neurology [@Khnel2021], cardiovascular disease [@KassahunYimer2020], or infection diseases [@Wu2007].

The current landscape of open-source software allowing end-users to easily fit and use joint models consists primarily of R packages such as JMbayes [@JMbayes], rstanarm [@rstanarm], joineR [@joineR], and JM [@JM]. These packages typically limit the longitudinal model to linear forms (for the parameters) preventing users from fitting joint models with saturating biological processes resulting in nonlinear profiles for the biomarker.
In contrast, the present software supports fitting any longitudinal and survival model, provided the joint model fulfills continuity and certain smoothness characteristics required for numerical procedures. Such flexibility is crucial for studying complex biological processes.

# Formulation


To build a joint model, we augment the survival analysis hazard function in time $h(t) = \lim_{\delta \to 0} P(t\leq T\leq t+\delta | T \geq t)/\delta$ for a positive random variable $T$, by incorporating a link $l$ to a longitudinal process. The longitudinal process is modeled by a function $m:\mathbb{R} \to \mathbb{R}$, for example a non-linear mixed effects model [@Wu2007]. Let the function $h_0:\mathbb{R} \to \mathbb{R}_{\geq 0}$ describe a baseline hazard and $\gamma\in\mathbb{R}$ be a coefficient of the link contribution. The hazard of the joint model is

$$ h(t) = h_0(t) \exp(\gamma\cdot l(m(t))),$$

where $l$ is the link to the longitudinal model. Some examples of links $l(m(t)) = (l \circ m)(t)$ are given in @Rizopoulos2012 such as the derivative $d/dt \; m(t)$ or integral $\int_0^t m(u) \, du$ operators.


Now we extend this idea to multiple longitudinal models. Suppose that we have $k\in \mathbb{N}$ longitudinal models $\{m_{1},\dots, m_{k}\}$ as well as $k$ link functions $\{l_{1},\dots, l_{k}\}$. Let $M: \mathbb{R} \to \mathbb{R}^k$ and $L:\mathbb{R}^k \to \mathbb{R}^k$ be the vector of functions

$$
    M(t) \mapsto \begin{pmatrix}
    m_{1}(t) \\ m_{2}(t) \\ \dots \\ m_{k}(t)
\end{pmatrix} \text{, }
    L\begin{pmatrix}
    \mu_1 \\ \mu_2 \\ \dots \\ \mu_k
\end{pmatrix} \mapsto \begin{pmatrix}
    l_1(\mu_1) \\ l_2(\mu_2) \\ \dots \\ l_k(\mu_k)
\end{pmatrix} 
$$
and together
$$L(M(t)) =\begin{pmatrix}
    l_1(m_{1}(t)) \\ l_2(m_{2}(t)) \\ \dots \\ l_k(m_{k}(t))
\end{pmatrix}.$$

For the link vector $L(M(t))$ we consider the coefficient vector $\gamma \in \mathbb{R}^k$ and formulate the hazard as follows

$$h(t) = h_0(t) \exp\left(\sum_{j = 1}^{k}\gamma_{j} l_j(m_{j}(t))  \right) = h_0(t) \exp(\gamma^\top \cdot L(M(t))).$$

In addition, we consider covariates $x\in \mathbb{R}^p, p\in\mathbb{N}$ and coefficients $\beta \in \mathbb{R}^p$. This results in the hazard
$$h(t) = h_0(t) \exp\left(\gamma^\top \cdot L(M(t)) +  \beta^\top \cdot x \right).$$

The probability density function in survival analysis can be described using the hazard function
$$f(t) = h(t) \exp\left(-\int_0^t h(u) \, du\right).$$
Note for joint models with links to nonlinear longitudinal models $\int_0^t h(u) \, du$ generally does not have a closed form, thus numerical integration is required.


## Likelihood calculations

Suppose that we have $n\in \mathbb{N}$ individuals. For each individual $i\in \{1,\dots, n\}$ we observe $n_i \in \mathbb{N}$ different longitudinal measurements $\{y_{i1}, \dots, y_{in_i}\}\subseteq \mathbb{R}$ at associated time points $\{t_{i1}, \dots, t_{in_i}\}\subseteq \mathbb{R}$ at which the measurements were recorded. In addition, we measure an event time $\tau_i \in \mathbb{R}$ and an event indicator $\delta_i \in \{0,1\}$. Without loss of generality, we will consider right-censored data; this can be adapted to other censoring processes. Let $Y_i := (\tau_i,\delta_i,(y_{i1}, \dots, y_{in_i}),(t_{i1}, \dots, t_{in_i}))$ be the measurements associated with individual $i$ and $Y = \{Y_1, \dots, Y_n\}$ all observations.




Let $\theta_H$ describe the parameters for the baseline hazard, $\theta_J$ for the joint model, and $\theta_L$ for longitudinal models. The likelihood of the joint model is comprised of the likelihood of the survival measurements and the longitudinal measurements.

$$\log L(Y | (\theta_H, \theta_J, \theta_L)) = \sum_{i=1}^{n} \log ( L((\tau_i, \delta_i) |  (\theta_H, \theta_J, \theta_L))) +  \sum_{i = 1}^{n}\sum_{ j=1}^{n_i} \log( L(t_{ij},y_{ij} | \theta_L) )$$

For individual $i$ let $f_i$ be the joint probability density function and $S_i$ the survival function. The likelihood depends on the censoring process, for example for right-censored measurements $(\tau_i, \delta_i)$ is given by

$$\log ( L((\tau_i, \delta_i) |  (\theta_H, \theta_J, \theta_L))) = \delta_i \log(f_i(\tau_i)) - (1-\delta_i)\int_0^{\tau_i} h_i(u) \,du$$

For the longitudinal model, the likelihood depends on the error process you use. Let $p_{m_i(t_{ij})}$ be the probability density function for measurements for model $m_i$ at time $t_{ij}$ for a given error, for example, the additive error or a multiplicative error. Then the longitudinal likelihood is given by

$$\log( L(t_{ij},y_{ij} | \theta_L)) = \log(p_{m_i(t_{ij})}(y_{ij}))$$






# Example


The following example showcases the simplicity and similarity to the mathematical description of the model that is achieved for the modeling of non-linear joint models using `JointSurvivalModels.jl`. It follows the simulation study by [@Kerioui2020]. They specify a longitudinal model for $\Psi = (\text{BSLD}, g, d, \phi) \in \mathbb{R}^4$ as


$$\text{SLD}(t,\Psi) = \begin{cases}
    \text{BSLD}\exp(gt) & t < t_x \\
    \text{BSLD}\exp(gt_x) (\phi \exp(-d(t-t_x)) + (1-\phi)\exp(g(t-t_x))) & t \geq t_x.
\end{cases}$$

In code:

```julia
function sld(t, Ψ, tx = 0.0)
    BSLD, g, d, φ = Ψ
    Δt = t - tx
    if t < tx
        return BSLD * exp(g * t)
    else
        return BSLD * exp(g * tx) * (φ * exp(-d * Δt) + (1 - φ) * exp(g * Δt))
    end
end
```

For the survival distribution, they use an Exponential distribution which has a constant hazard $h_0(t) = 1 / \lambda$ with scale parameter $\lambda \in \mathbb{R}_{\geq 0}$

```julia
h_0(t, λ) = 1/λ
```

In the mixed effects model every individual $i$ has a different mixed effects parameter denoted by $\Psi_i$ which defines the longitudinal model $M_i(t) = \text{SLD}(t,\Psi_i)$ resulting in the joint hazard for individual $i$

$$h_i(t) = h_0(t) \exp(\gamma \cdot L(M_i(t))) = h_0(t) \exp(\gamma * \text{id}(\text{SLD}(t, \Psi_i))).$$

The identity $id$ was used as a link. In code, the distribution of the joint model defined by this hazard is given by:
```julia
my_jm(κ, λ, γ, Ψ, tx) = JointSurvivalModel(t -> h_0(t, κ, λ), γ, t -> sld(t, Ψ, tx))
```

The mixed effects model contains population parameters $\mu = (\mu_{\text{BSLD}},\mu_d, \mu_g, \mu_\phi)$ and random effects $\eta_i = (\eta_{\text{BSLD},i},\eta_{d,i}, \eta_{g,i}, \eta_{\phi,i})$ which are normally distributed around zero $\eta_i \sim N(0, \Omega), \Omega = \text{diag}(\omega_{\text{BSLD}}^2,\omega_d^2, \omega_g^2, \omega_\phi^2)$. For $\text{BSLD}, g, d$ a log-normal transform $\log(\Psi_{q,i}) = log (\mu_q) + \eta_{q,i},\, q\in \{\text{BSLD}, g, d\}$ was used while for $\phi$ a logit transform $\text{logit}(\Psi_{\phi,1}) = \text{logit}(\mu_\phi) + \eta_{\phi,1}$ was used.

With this information, a Bayesian model can be specified in `Turing.jl` [@Turing.jl] by giving prior distributions for the parameters and calculations for the likelihood. To calculate the likelihood of the survival time and event indicator the `JointModel` is used. This results in a canonical translation of the statistical ideas into code. For longitudinal data, a multiplicative error model is used using $e_{ij} \sim N(0, \sigma^2)$ given by $y_{ij} = \text{SLD}(t_{ij},\Psi_i)(1+e_{ij})$ is used. The model and prior setup from @Kerioui2020 can be implemented as follows in code:

```julia
@model function identity_link(
    longit_ids,
    longit_times,
    longit_measurements,
    surv_ids,
    surv_times,
    surv_event,
)
    # treatment start at study start
    tx = 0.0
    # number of longitudinal and survival measurements
    n = length(surv_ids)
    m = length(longit_ids)
    # ---------------- Priors -----------------
    ## priors longitudinal
    # population parameters
    μ_BSLD ~ LogNormal(3.5, 1)
    μ_d ~ Beta(1, 100)
    μ_g ~ Beta(1, 100)
    μ_φ ~ Beta(2, 4)
    μ = (BSLD = μ_BSLD, d = μ_d, g = μ_g, φ = μ_φ)
    # standard deviation of random effects
    ω_BSLD ~ LogNormal(0, 1)
    ω_d ~ LogNormal(0, 1)
    ω_g ~ LogNormal(0, 1)
    ω_φ ~ LogNormal(0, 1)
    Ω = (BSLD = ω_BSLD^2, d = ω_d^2, g = ω_g^2, φ = ω_φ^2)
    ## η describing the random effects
    η_BSLD ~ filldist(Normal(0, Ω.BSLD), n)
    η_d ~ filldist(Normal(0, Ω.d), n)
    η_g ~ filldist(Normal(0, Ω.g), n)
    η_φ ~ filldist(Normal(0, Ω.φ), n)
    # Transforms
    Ψ = [  [μ_BSLD * exp(η_BSLD[i]),
            μ_g * exp(η_g[i]),
            μ_d * exp(η_d[i]),
            inverse(logit)(logit(μ_φ) + (η_φ[i])),] 
         for i = 1:n]
    # multiplicative error
    σ ~ LogNormal(0, 1)
    ## prior survival
    λ ~ LogNormal(5, 1)
    ## prior joint model
    γ ~ truncated(Normal(0, 0.5), -0.1, 0.1)
    # ---------------- Likelihoods -----------------
    # add the likelihood of the longitudinal process
    for data in 1:m
        id = Int(longit_ids[data])
        meas_time = longit_times[data]
        sld_prediction = sld(meas_time, Ψ[id], tx)
        if isnan(sld_prediction) || sld_prediction < 0
            sld_prediction = 0
        end
        longit_measurements[data] ~ Normal(sld_prediction, sld_prediction * σ)
    end
    # add the likelihood of the survival model with link
    baseline_hazard(t) = h_0(t, λ)
    for i = 1:n
        id = Int(surv_ids[i])
        id_link(t) = sld(t, Ψ[id], tx)
        censoring = Bool(surv_event[id]) ? Inf : surv_times[id]
        # here we use the JointSurvivalModel
        surv_times[i] ~ censored(
            JointSurvivalModel(baseline_hazard, γ, id_link),
            upper = censoring
        )
    end
end
```
When sampling the posterior the log probability density function `logpdf` implemented in `JointModels.jl` is called for a distribution given specific parameters. The numerical calculation of the likelihood is then used in the sampling process. Sampling with the `Turing.Inference.NUTS` algorithm 2'000 posterior samples using 1'000 burn-in samples results in posterior statistic:

```
parameters        mean        std      mcse       rhat  |  ture_parameters    
    Symbol     Float64    Float64   Float64    Float64  |       

    μ_BSLD     61.3971     4.4043    0.2737     1.0021  |        60
       μ_d      0.0058     0.0014    0.0001     0.9998  |         0.0055 
       μ_g      0.0018     0.0002    0.0000     0.9998  |         0.0015
       μ_φ      0.1729     0.0591    0.0049     1.0001  |         0.2
         σ      0.1767     0.0057    0.0001     1.0008  |         0.18
         λ   1491.2720   307.5059    6.4314     1.0008  |      1450
         γ      0.0103     0.0012    0.0000     1.0005  |         0.01
```
Notice that the link coefficient $\gamma$ was sampled around the true value $0.01$ with a small variance. The survival and population parameters are well represented by the posterior samples indicated by the $\hat r$ value close to one and the relative closeness of the mean to the true parameter.


Additionally, `JointModels.jl` implements the generation of random samples of a joint distribution. This allows the creation of posterior predictive checks, simulations, or individual predictions \autoref{fig:ind_pred} using `Turing.jl`, which are a major step in a Bayesian workflow when validating a model [@BayesianWorkflow]. 

![The figure showcases individual posterior prediction for the mixed effects longitudinal sub-model alongside the longitudinal observations. The figure also shows the individual survival predictions conditioned on survival until the last measurement based on the joint survival distribution. The light-colored ribbons represent the 95% quantile of the posterior predictions while the line represents the median. To create the posterior predictions the above `Turing.jl` model was used. \label{fig:ind_pred}](individual_prediction.svg){ width=80% }






# Acknowledgements

This software was initially developed at Hoffmann-La Roche Ltd. before transitioning into an open source project. Professor Peter Bühlmann from ETH Zürich facilitated the development and open source project.


# References

