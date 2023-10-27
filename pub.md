# JOSS Pub

**Scope**: Present the software. What are the application/needs, on what theoretical basis is it build on, what other options exist / don’t exist.

**Size**: Publications are short. Starting form a single page. Can include plots and code snippets.

> The paper should be between 250-1000 words. Authors submitting papers significantly longer than 1000 words may be asked to reduce the length of their paper.

### ToC
Summary and statement of need seem to be interchangable. Staement of need could serve as an introduction
- Statement of need
  - Describe the application of joint models in clinical research and oncology (sources??)
  - Motivation nonlinear joint models (sources?? **Why nonlinear is better than linear models in oncology?**) and describe their difficulty of implementation. need for bayesian sampling
  - Benefit of julia ecosystem for likelihood calculation and sample generation in Bayesian frameworks.
    - Introduce Turing and the flexibility it comes with, using the whole julia language and interoperability with the julia ecosystem
    - Mention Distributions.jl, Quad-GK implementation and the DiffEQ ecosystem as key for implementing joint models in a general framework
- Summary
  - Present joint model formulation with minimal assumptions on longitudinal submodel
    - Mention the options for the link to the longitudinal process and how it acts on the longitudinal model.
    - State that the link & longitudinal model should have low *oscillation* (?? name for condition for numerical integratoin ??) for numerical methods
  - Extend to multiple longitudinal processes and give likelihood calculation
  - Describe usage of bayesian algorithms to sample posterior distribution given numerical likelihood calculation.
- Example
  - Replication of spesific joint model, e.g. Kerioui et al. or simpler non linear joint model (any reference?)
  - Some nice plots and core code snippets
- Related Software
  - Quick overview over joint model software in R (jm, jmbayes), mention that existing solutions are not capable of working with non-linear models
  - Stan implementation from paper, give small comparison to showcare computation speed and similar convergence of the algorithms
 
### Published Papers
Longer julia publications:
- https://joss.theoj.org/papers/10.21105/joss.05669
- https://joss.theoj.org/papers/10.21105/joss.05187
- https://joss.theoj.org/papers/10.21105/joss.05786

Clinical statistics publications:
- https://joss.theoj.org/papers/10.21105/joss.05345
- https://joss.theoj.org/papers/10.21105/joss.02816

### Guidelines

Docs for submissions can be found here: **https://joss.readthedocs.io/en/latest/submitting.html**
