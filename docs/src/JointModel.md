# General Joint Model


This struct implements a general family of joint models. Let $h_0:\mathbb{R} \to\mathbb{R}_{+}$ be a baseline hazard function. Suppose we have $k\in \mathbb{N}$ longitudinal models $\{m_{1},\dots, m_{k}\}\subset \{\text{Functions: }\mathbb{R}\to\mathbb{R}\}$ as well as $k$ links $\{l_{1},\dots, l_{k}\}\subset \{\text{Operators on functions: }\mathbb{R}\to\mathbb{R}\}$. Let $M: \mathbb{R} \to \mathbb{R}^k$ and $L:\mathbb{R}^k \to \mathbb{R}^k$ be the multidimensional vector versions

```math
\begin{align*}
    M(t) \mapsto \begin{pmatrix}
    m_{1}(t) \\ m_{2}(t) \\ \dots \\ m_{k}(t)
\end{pmatrix} \text{, }
    L\begin{pmatrix}
    \mu_1 \\ \mu_2 \\ \dots \\ \mu_k
\end{pmatrix} \mapsto \begin{pmatrix}
    l_1(\mu_1) \\ l_2(\mu_2) \\ \dots \\ l_k(\mu_k)
\end{pmatrix} \text{ and } L(M(t)) =\begin{pmatrix}
    l_1(m_{1}(t)) \\ l_2(m_{2}(t)) \\ \dots \\ l_k(m_{k}(t))
\end{pmatrix}.
\end{align*}
```


In code $L(M(t))$ corresponds to an array of unary functions (one argument). You are responsible for choosing the longitudinal model and link and finally the application of the link to the longitudinal model. For the link model vector $L(M(t))$ we consider a coefficient vector $\gamma \in \mathbb{R}^k$. Then we can formulate a hazard as follows

```math
h(t) = h_0(t) \exp\left(\sum_{j\in [k]}\gamma_{j} l_j(m_{j}(t))  \right) = h_0(t) \exp(\gamma' \cdot L(M(t))).
``` 
Additionally we consider covariates $x\in \mathbb{R}^l, l\in\mathbb{N}$ and coefficients $\beta \in \mathbb{R}^l$. This results in the hazard

```math
\begin{align*}
h(t) &= h_0(t) \exp\left(\gamma' \cdot L(M(t)) + \beta' \cdot x\right)\\
     &= h_0(t) \exp\left(\sum_{j\in [k]}\gamma_{j} l_j(m_{j}(t)) + \sum_{j\in [l]} x_j  \beta_j  \right),
\end{align*}
```
which is implemented in the general joint model:


```@docs
JointModels.JointModel
```

Its hazard is calculated by:

```@docs
JointModels.hazard(jm::JointModel, t::Real)
```