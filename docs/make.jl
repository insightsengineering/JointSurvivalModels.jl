using Documenter
using JointModels
using Random
using Distributions

makedocs(
    sitename = "JointModels",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "First Example" => "FirstExample.md",
        "API" => Any[
            "JointModel" => "JointModel.md",
            "HazardBasedDistribution" => "HazardBasedDistribution.md"
        ],
    ],
    modules = [JointModels]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/insightsengineering/JointModels.jl.git",
)
