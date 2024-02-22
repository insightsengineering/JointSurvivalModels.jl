using Documenter
using JointSurvivalModels
using Random
using Distributions

makedocs(
    sitename = "JointSurvivalModels",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "First Example" => "FirstExample.md",
        "API" => Any[
            "JointSurvivalModel" => "JointSurvivalModel.md",
            "HazardBasedDistribution" => "HazardBasedDistribution.md"
        ],
    ],
    modules = [JointSurvivalModels]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/insightsengineering/JointSurvivalModels.jl.git",
)
