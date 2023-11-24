using Pkg

Pkg.activate(".")
Pkg.instantiate()

using Documenter
using JointModels
using Random
using Distributions

makedocs(
    sitename = "JointModels",
    format = Documenter.HTML(size_threshold = nothing),
    modules = [JointModels]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
