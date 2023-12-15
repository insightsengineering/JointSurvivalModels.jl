# First Examples

## Joint Models
The struct `GeneralJointModel` allows you to implement joint models. Let us consider a simple example. First we describe a model and generate data. We define a nonlinear mixed-effects longitudinal model

```math
m_i(t) = a_i + b * t^(c_i) * cos(d_i * t)^2,
```
where ``a_i, c_i`` and ``d_i`` are mixed effects while ``b`` is a population parameter. We assume distributions

```math
a_i ~ Uniform(20,200)
b ~ Exponential(1.5)
c_i ~ Beta(2,5)
d_i ~ Normal(0,1)
```

```julia
n = 100
dist1 = Uniform(20,200)
dist2 =  Exponential(1.5)
dist3 =  Beta(2,5)
dist4 = Normal(0,1)
a = rand(dist1, n)
b = rand(dist1)
c = rand(dist1, n)
d = rand(dist1, n)

m(t, i) = a[i] + b * t^(c[i]) * cos(d[i] * t)^2


```



