function enslvm_spectra(X, C; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

    rg = MersenneTwister(seed)

    N = length(X); @assert(N == length(C))
    D = length(X[1])

    # randomly initialise network parameters
    
    W = let 

        t(x) = softmax(x, dims=1)

        f = net(Q = Q, M = M, D = D, o = t)
        
        [randn(rg, numparam(f)) for _ in 1:K]

    end


    # randomly initialise latent

    Z = [randn(Q) for n in 1:N]
    
    # initialise scalings

    scales = ones(N)

    enslvm_spectra(X, C, W, Z, scales; labels=labels, K = K, M = M, Q = Q, iterations = iterations, seed = seed, α = α, η = η)

end


function enslvm_spectra(X, C, W, Z, scales; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

    N = length(X); @assert(N == length(C))
    D = length(X[1])

    # report

    @printf("Running enslvm_spectra with K=%d and data of %d number of data items of dimension %d\n", K, N, D)


    # define ensemnle output

    t(x) = softmax(x, dims=1)

    f = net(Q = Q, M = M, D = D, o = t)
    
    F(W, z) = mapreduce(w -> f(w, z), +, W) / K
    
    B = bootstrapweights(K, N; rg = MersenneTwister(seed))

    function pack(Zₙ, sₙ) 

        [vec(Zₙ); sqrt(sₙ)]

    end

    function unpack(p) 

        p[1:end-1], p[end]^2

    end


    #-------------------------------------------
    function objective_Zₙ(g, Zₙ, sₙ, n)
    #-------------------------------------------

        @assert(length(Zₙ) == Q)

        local aux = zero(eltype(Zₙ))

        for k in 1:K

            diff = X[n] - sₙ * g[k](Zₙ)

            aux += B[k][n] * sum(abs2.(diff) ./ C[n])
            
        end

        # penalty on coordinates

        aux += η*sum(abs2.(Zₙ))

        aux

    end


    #-------------------------------------------
    function objective_single_member(w, Z, s, b)
    #-------------------------------------------

        local aux = zero(eltype(w))

        local gₖ = f(w)
        
        for n in 1:N

            diff = X[n] - s[n] * gₖ(Z[n])

            aux += b[n] * sum(abs2.(diff) ./ C[n])

        end

        aux += α*sum(abs2.(w))

        return aux

    end


    #-------------------------------------------
    function ensemble_performance(W, Z, s)
    #-------------------------------------------

        local aux = 0.0

        for k in 1:K, n in 1:N

            diff = X[n] - s[n] * f(W[k], Z[n])

            aux += sum(abs2.(diff) ./ C[n])

        end

        return aux / K

    end


    for iter in 1:iterations

        #----------------------------------------------------
        # hold ensemble fixed, optimise latent coordinates Z
        #----------------------------------------------------
        
        let 
          
            pr = Progress(N, desc = "Latent coordinates")
          
            optZ = Optim.Options(iterations = 5, show_trace = false, show_every=1)
       
            local g = map(f, W)

            Threads.@threads for n in 1:N
            
                let

                    obj(x) = objective_Zₙ(g, unpack(x)..., n)

                    grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

                    result = optimize(obj, grad!, pack(Z[n], scales[n]), LBFGS(), optZ)
                    
                    Z[n], scales[n] = unpack(result.minimizer)

                end

                next!(pr)

            end

        end


        
        #-----------------------
        # hold latent fixed
        #-----------------------

        pr = Progress(K, desc = "Ensemble members")

        optW = Optim.Options(iterations = 3, show_trace = false, show_every=1)
    
        Threads.@threads for k in 1:K

            let

                obj(w) = objective_single_member(w, Z, scales, B[k])
                
                grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

                W[k] = optimize(obj, grad!, W[k], LBFGS(), optW).minimizer
                
            end
            next!(pr)

        end

        @printf("Iteration %d, ensemble performance is %f\n", iter, ensemble_performance(W, Z, scales))

    end

    z -> F(W, z), W, Z, scales
    
end


