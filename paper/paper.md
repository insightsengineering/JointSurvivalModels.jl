---
title: 'JointModels.jl: A Julia package for general Bayesian joint models'
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
This Julia package implements a numerical approach to define a distribution based on the survival hazard function. In particular this provided a mechanism of defining joint models of time-to-even data and longitudinal measurements. Using numerical integration, the likelihood of events can be calculated, allowing Bayesian inference frameworks to sample the posterior distribution of the model's parameters. Additionally, this implementation is able to generate samples of joint models. This allows its use in simulations, which are common in Bayesian workflows [@BayesianWorkflow].


# Statement of need

Over the last decade, there has been a growing interest in joint models for longitudinal and time-to-event outcome data. This is the case in clinical research where biomarkers are often measured repeatedly over time, with the expectation that they may provide insights into the likelihood of a long-term clinical outcome such as disease progression or adverse events. As a consequence, joint models have been proposed to leverage this data and used for the prediction of individual risks, the evaluation of surrogacy of a biomarker for a clinical outcome, and for making inferences about treatment in various therapeutic areas.

In oncology, it is well known that the individual risk of death depends on the treatment-induced changes in tumor size over time [@Tardivon2019]. Because pharmacologic effects are typically transient, nonlinear mixed-effect models are required to capture central tendency and inter-individual variability in tumor size, while parametric models can be used for analysing time to death. Joint models have been used in other therapeutic areas such as neurology [@Khnel2021], cardiovascular disease [@KassahunYimer2020], or infection diseases [@Wu2007].

The current landscape of open-source software allowing end-users to easily fit and use joint models consists primarily of R packages such as JMbayes [@JMbayes], rstanarm [@rstanarm], joineR [@joineR], or JM [@JM]. These packages typically limit the longitudinal model to linear forms (for the parameters) preventing users from fitting joint models with saturating biological processes resulting in nonlinear profiles for the biomarker.
In contrast, the present software supports fitting any longitudinal and survival model, provided the joint model fulfills continuity and certain smoothness characteristics for numerical procedures. Such flexibility is crucial for studying complex biological processes.

# Formulation


To build a joint model, we augment the survival analysis hazard function $h(t) = \lim_{\delta \to 0} P(t\leq T\leq t+\delta | T \geq t)/\delta$ for a positive random variable $T$, by incorporating a link $l$ to a longitudinal process. The longitudinal process is modeled by a function $m:\mathbb{R} \to \mathbb{R}$, for example a non-linear mixed effects model [@Wu2007]. Let the function $h_0:\mathbb{R} \to \mathbb{R}_{\geq 0}$ describe a baseline hazard and $\gamma\in\mathbb{R}$ be a coefficient of the link contribution. The hazard of the joint model is

$$ h(t) = h_0(t) \exp(\gamma\cdot l(m(t))).$$

Where $l$ is the link to the longitudinal model. In general, the link is an operator on the longitudinal models. Some examples of links $l(m(t)) = (l \circ m)(t)$ are given in @Rizopoulos2012 such as the derivative $d/dt \; m(t)$ or integral $\int_0^t m(u) \, du$ operators.


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

For the link vector $L(M(t))$ we consider the coefficient vector $\gamma \in \mathbb{R}^k$. Let $[k] := \{1,\dots, k\}$ then we formulate the hazard as follows.

$$h(t) = h_0(t) \exp\left(\sum_{j\in [k]}\gamma_{j} l_j(m_{j}(t))  \right) = h_0(t) \exp(\gamma^\top \cdot L(M(t))).$$

In addition, we consider covariates $x\in \mathbb{R}^p, p\in\mathbb{N}$ and coefficients $\beta \in \mathbb{R}^l$. This results in the hazard
$$h(t) = h_0(t) \exp\left(\gamma^\top \cdot L(M(t)) +  \beta^\top \cdot x \right).$$

The probability density function in survival analysis can be described by
$$f(t) = h(t) \exp\left(-\int_0^t h(u) \, du\right).$$
Note for nonlinear longitudinal models $\int_0^t h(u) \, du$ generally does not have a closed form, thus numerical integration is required.


## Likelihood calculations

Suppose that we have $n\in \mathbb{N}$ individuals. For each individual $i\in [n]=\{1,\dots, n\}$ we observe $n_i \in \mathbb{N}$ different longitudinal measurements $\{y_{i1}, \dots, y_{in_i}\}\subseteq \mathbb{R}$ at associated time points $\{t_{i1}, \dots, t_{in_i}\}\subseteq \mathbb{R}$ at which the measurements were recorded. In addition, we measure an event time $\tau_i \in \mathbb{R}$ and an event indicator $\delta_i \in \{0,1\}$. Without loss of generality we will consider right-censored data; this can be adapted to other censoring processes. Let $Y_i := (\tau_i,\delta_i,(y_{i1}, \dots, y_{in_i}),(t_{i1}, \dots, t_{in_i}))$ be the measurements associated with individual $i\in [n]$ and $Y = \{Y_1, \dots, Y_n\}$ all observations.




Let $\theta_H$ describe the parameters for the baseline hazard, $\theta_J$ for the joint model, and $\theta_L$ for longitudinal models. The likelihood of the joint model is comprised of the likelihood of the survival measurements and the longitudinal measurements.

$$\log L(Y | (\theta_H, \theta_J, \theta_L)) = \sum_{i\in[n]} \log ( L((\tau_i, \delta_i) |  (\theta_H, \theta_J, \theta_L))) +  \sum_{i\in[n]}\sum_{ j\in[n_i]} \log( L(t_{ij},y_{ij} | \theta_L) )$$

For individual $i\in[n]$ let $f_i$ be the joint probability density function and $S_i$ the survival function. The likelihood depends on the censoring process, for example for right-censored measurements $(\tau_i, \delta_i)$ is given by

$$\log ( L((\tau_i, \delta_i) |  (\theta_H, \theta_J, \theta_L))) = \delta_i \log(f_i(\tau_i)) - (1-\delta_i)\int_0^{\tau_i} h_i(u) \,du$$

For the longitudinal model, the likelihood depends on the error process you use. Let $p_{m_i(t_{ij})}$ be the probability density function for measurements for model $m_i$ at time $t_{ij}$ for a given error, for example, the additive error or a multiplicative error. Then the longitudinal likelihood is given by

$$\log( L(t_{ij},y_{ij} | \theta_L)) = \log(p_{m_i(t_{ij})}(y_{ij}))$$






# Example

The following example showcases the simplicity and similarity to the mathematical description of the model that is achieved for the modeling of non-linear joint models using `JointModels.jl`. The code can be found in the [example](https://github.com/insightsengineering/JointModels.jl/tree/main/example) folder in the project repository. Following @Kerioui2020 a longitudinal model for the sum of longest diameters $\text{SLD}: \mathbb{R} \to \mathbb{R}_{\geq 0}$ is specified with parameters  $\Psi = (\text{BSLD}, g, d, \phi)$ where $\text{BSLD}, g, d \in \mathbb{R}_{\geq 0},\; \phi \in [0,1]$ and start of treatment $t_x$

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

For the survival distribution they use an Exponential distribution which has a constant hazard $h_0(t) = 1 / \lambda$ with scale parameter $\lambda \in \mathbb{R}_{\geq 0}$

```julia
h_0(t, λ) = 1/λ
```

In the mixed effects model every individual $i$ has a different parameter denoted by $\Psi_i$ resulting in the joint hazard

$$h_i(t) = h_0(t) \exp(\gamma \cdot L(M(t))) = h_0(t) \exp(\gamma * \text{id}(\text{SLD}(t, \Psi_i))).$$

The identity $id$ was used as a link. In code the distribution of the joint model defined by this hazard is given by:
```julia
my_jm(λ, γ, Ψ_i, tx) = JointModel(t -> h_0(t, λ), γ, t -> sld(t, Ψ_i, tx))
```

The mixed effects model contains population parameters $\mu = (\mu_{\text{BSLD}},\mu_d, \mu_g, \mu_\phi)$ and random effects $\eta_i = (\eta_{\text{BSLD},i},\eta_{d,i}, \eta_{g,i}, \eta_{\phi,i})$ which are normally distributed around zero $\eta_i \sim N(0, \Omega), \Omega = \text{diag}(\omega_{\text{BSLD}}^2,\omega_d^2, \omega_g^2, \omega_\phi^2)$. For $\text{BSLD}, g, d$ a log-normal transform $\log(\Psi_{g,i}) = log (\mu_g) + \eta_{g,i}$ was used while for $\phi$ a logit transform $\text{logit}(\Psi_{\phi,1}) = \text{logit}(\mu_\phi) + \eta_{\phi,1} $ was used.

With this information, a Bayesian model can be specified in `Turing.jl` [@Turing] by giving prior distributions for the parameters and calculations for the likelihood. To calculate the likelihood of the survival time and event indicator the software is used. This results in a canonical translation of the statistical ideas into code. For longitudinal data, a multiplicative error model is used using $e_{ij} \sim N(0, \sigma^2)$ given by $y_{ij} = \text{SLD}(t_{ij},\Psi_i)(1+e_{ij})$ is used. The model and prior setup from @Kerioui2020 can be implemented as follows in code:

```julia
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
        # here we use the JointModel
        try
            surv_times[i] ~
                censored(JointModel(baseline_hazard, γ, id_link), upper = censoring)
        catch
        end
    end
end
```
When sampling the posterior the log probability density function of the joint model is called conditioned on specific parameters. The numerical calculation of the likelihood is then used in the sampling process. Sampling with the `Turing.Inference.NUTS` algorithm 400 posterior samples using 200 burn-in results in posterior statistic:

```
parameters        mean        std      mcse       rhat  |  ture_parameters    
    Symbol     Float64    Float64   Float64    Float64  |       

    μ_BSLD     62.1952     4.8282    0.7664     1.0096  |        60
       μ_d      0.0056     0.0015    0.0002     1.0473  |         0.0055 
       μ_g      0.0017     0.0003    0.0000     0.9986  |         0.0015
       μ_φ      0.1807     0.0615    0.0079     1.0066  |         0.2
         σ      0.1768     0.0063    0.0006     1.0283  |         0.18
         λ   1393.6532   286.6294   15.9133     0.0059  |      1450
         γ      0.0103     0.0012    0.0001     0.0005  |         0.01
```
Notice that the link coefficient $\gamma$ was sampled around the true value with a narrow variance. In general the survival parameters are well represented by the posterior samples. The mixed effects model parameters as well indicated by the $\hat r$ value which is close to one, with the potential exception of $\mu_d$.


Additionally, `JointModels.jl` implements the generation of random samples of a joint distribution which enables `Turing.jl` to sample a joint distribution. This allows to create posterior predictive checks, simulations or individual predictions \autoref{fig:ind_pred}, which are a major step in a Bayesian workflow when validating a model [@BayesianWorkflow]. 

![The figure showcases individual posterior prediction for the mixed effects longitudinal sub model along side the longitudinal observations. The figure also shows the individual survival predictions conditioned on survival until the last measurement based on the joint survival distribution. The light colored ribbons represent the 95% quantile of the posterior predictions while the line represents the median. To create the posterior predictions the above `Turing.jl` model was used. \label{fig:ind_pred}](individual_prediction.svg){ width=80% }






# Acknowledgements

# References

