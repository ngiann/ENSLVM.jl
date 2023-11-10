function enslvm_spectra2(X, C; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

    rg = MersenneTwister(seed)

    D, N = size(X); @assert(size(X) == size(C))
    

    # randomly initialise network parameters
    
    W = let 

        t(x) = softmax(x, dims=1)

        f = net(Q = Q, M = M, D = D, o = t)
        
        [randn(rg, numparam(f)) for _ in 1:K]

    end


    # randomly initialise latent

    Z = randn(rg, Q, N)
    
    # initialise scalings

    scales = ones(N)

    enslvm_spectra2(X, C, W, Z, scales; labels=labels, K = K, M = M, Q = Q, iterations = iterations, seed = seed, α = α, η = η)

end


function enslvm_spectra2(X, C, W, Z, scales; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

    D, N = size(X); @assert(size(X) == size(C))

    # report

    @printf("Running enslvm_spectra2 with K=%d and data of %d number of data items of dimension %d\n", K, N, D)


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
    function objective_Zₙ(g, Xₙ, Cₙ, Zₙ, sₙ, b)
    #-------------------------------------------

        @assert(length(Zₙ) == Q)

        local aux = zero(eltype(Zₙ))

        for k in 1:K

            diff = Xₙ - sₙ * g[k](Zₙ)

            aux += b[k] * sum(abs2.(diff) ./ Cₙ)
            
        end

        # penalty on coordinates

        aux += η*sum(abs2.(Zₙ))

        aux

    end


    #-------------------------------------------
    function objective_single_member(w, Z, s, b)
    #-------------------------------------------

        local gₖ = f(w)
        
        local diff = X - gₖ(Z)*Diagonal(s)

        sum((abs2.(diff) ./ C)*Diagonal(b)) + α*sum(abs2.(w))

    end


    #-------------------------------------------
    function ensemble_performance(W, Z, s)
    #-------------------------------------------

        local aux = 0.0

        for k in 1:K
            
            diff = X - f(W[k], Z)*Diagonal(s)

            aux += sum((abs2.(diff) ./ C)*Diagonal(B[k]))

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

                    obj(x) = objective_Zₙ(g, X[:,n], C[:,n], unpack(x)..., [B[k][n] for k in 1:K])

                    grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

                    result = optimize(obj, grad!, pack(Z[:,n], scales[n]), LBFGS(), optZ)
                    
                    Z[:,n], scales[n] = unpack(result.minimizer)

                end

                next!(pr)

            end

        end


        
        #-------------------------------------------------
        # hold latent fixed, optimise ensemble weights wₖ
        #-------------------------------------------------

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


