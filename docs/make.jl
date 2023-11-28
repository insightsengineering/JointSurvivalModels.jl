using Documenter
using JointModels
using Random
using Distributions

makedocs(
    sitename = "JointModels",
    format = Documenter.HTML(),
    modules = [JointModels]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/insightsengineering/jointmodels.jl.git",
)
