module ENSLVM

    using LinearAlgebra, JLD2, Zygote, Random, Printf, Optim, NNlib, ProgressMeter, Statistics

    include("net.jl")

    export net, numparam

    # include("enslvm.jl")

    # export enslvm

    include("bootstrapweights.jl")

    # include("enslvm_spectra2.jl")

    # export enslvm_spectra2

    # include("enslvm_spectra3.jl")

    # export enslvm_spectra3

    include("enslvm_spectra.jl") # ⟵ best so far

    export enslvm_spectra

    include("distill.jl")

    export distill

    include("getmodel.jl")

    export getmodel
    
end
