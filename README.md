
# Link Predictors against Adversarial Attacks

> For best experience, use in conjunction with [Revise.jl](https://github.com/timholy/Revise.jl)

Once loaded into Revise, revise Plotting.jl to suit your needs; so inside the REPL,

```julia
using GraphAttacks
p = Plotting
me = "Random Deletion" ; maps, aps, katzs = p.emit_evaluation_data("Scale Free (small)", me, budgets=[0 10 20 30 40 50], train_fraction=0.85)
p.print_for_pyplot([0 10 20 30 40 50], maps)
```

Play around with `p.main`, or `p.emit_evaluation_data` and `p.print_for_pyplot`, and the various variables inside `Plotting.jl`.

# Setup

See [https://tlienart.github.io/pub/julia/dev-pkg.html](https://tlienart.github.io/pub/julia/dev-pkg.html). Basically:

```julia
(@v1.4) pkg> dev GraphAttacks
```

# Development

```julia
(@v1.4) pkg> add Revise     # See https://timholy.github.io/Revise.jl/stable/config/
(@v1.4) pkg> add ExportAll
julia> using GraphAttacks
```

# References

** (2018) [Attack Tolerance of Link Prediction Algorithms: How to Hide Your Relations in a Social Network](https://arxiv.org/abs/1809.00152)

** Datasets: https://noesis.ikor.org/datasets/link-prediction

# Programming

- Julia: https://julialang.org/downloads/ (I'm using Julia 1.4.)
- LightGraphs: https://github.com/JuliaGraphs/LightGraphs.jl
