module ENSLVM

    using LinearAlgebra, Zygote, Random, Printf, PyPlot, Optim, NNlib, ProgressMeter, ForwardDiff, Statistics

    include("net.jl")

    export net, numparam

    include("enslvm.jl")

    export enslvm

    include("bootstrapweights.jl")

    include("enslvm_spectra2.jl")

    export enslvm_spectra2

    include("enslvm_spectra3.jl")

    export enslvm_spectra3

    include("enslvm_spectra.jl") # ⟵ best so far

    export enslvm_spectra

    include("ensemblehessian.jl")

    export ensemblehessian

end
