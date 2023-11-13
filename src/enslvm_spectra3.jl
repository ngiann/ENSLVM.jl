function enslvm_spectra3(X, C; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

    # fix random seed

    rg = MersenneTwister(seed)

    # sort out dimensions

    D, N = size(X); @assert(size(X) == size(C))
    

    #-------------------------------------------
    # Initialisations
    #-------------------------------------------
    
    # random weights

    W = let 

        t(x) = softmax(x, dims=1)

        f = net(Q = Q, M = M, D = D, o = t)
        
        [randn(rg, numparam(f)) for _ in 1:K]

    end


    # randomly latent

    Z = randn(rg, Q, N)
    

    # initialise scalings

    scales = ones(N)


    # call model with initialised parameters

    enslvm_spectra3(X, C, W, Z, scales; labels=labels, K = K, M = M, Q = Q, iterations = iterations, seed = seed, α = α, η = η)

end


function enslvm_spectra3(X, C, W, Z, scales; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

    # sort out dimensions

    D, N = size(X); @assert(size(X) == size(C))

    # report

    @printf("Running enslvm_spectra3 with K=%d and data of %d number of data items of dimension %d\n", K, N, D)


    # define ensemble output

    t(x) = softmax(x, dims=1)

    f = net(Q = Q, M = M, D = D, o = t)
    
    F(W, z) = mapreduce(w -> f(w, z), +, W) / K
    
    B = bootstrapweights(K, N; rg = MersenneTwister(seed))


    #-------------------------------------------
    function pack(Z, s) 
    #-------------------------------------------

        [vec(Z); sqrt.(s)]

    end


    #-------------------------------------------
    function unpack(p)
    #------------------------------------------- 

        
        MARK = 0

        local Z = reshape(p[MARK+1:MARK+N*Q], Q, N)

        MARK += N*Q

        local scales = p[MARK+1:MARK+N].^2

        MARK += N

        @assert(MARK == length(p))

        return Z, scales

    end


    #-------------------------------------------
    function objective_Z(W, Z, s)
    #-------------------------------------------

        local aux = zero(eltype(Z))

        for k in 1:K

            diff = X - f(W[k], Z)*Diagonal(s)

            aux += sum((abs2.(diff) ./ C)*Diagonal(B[k]))
            
        end

        # penalty on coordinates

        aux += η*sum(abs2.(Z))

        return aux

    end


    #-------------------------------------------
    function objective_single_member(w, Z, s, b)
    #-------------------------------------------

        local gₖ = f(w)
        
        local diff = X - gₖ(Z)*Diagonal(s)

        sum((abs2.(diff) ./ C)*Diagonal(b)) + α*sum(abs2.(w)) # penalty on weights

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
          
            optZ = Optim.Options(iterations = 5, show_trace = true, show_every=1)
       
            obj(x) = objective_Z(W, unpack(x)...)

            grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

            result = optimize(obj, grad!, pack(Z, scales), LBFGS(), optZ)
                    
            Z, scales = unpack(result.minimizer)

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


    #-------------------------------------------------
    # return results
    #-------------------------------------------------

    function predict(z)
        
        # local predₖ = map(w->f(w, z), W)
        
        # mean(predₖ), std(predₖ)

        F(W,z)

    end

    #predict(k,z) = f(W[k], z)
    
    predict, W, Z, scales, B
    
end


