function enslvm_spectra(X, C; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

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

    enslvm_spectra(X, C, W, Z, scales; labels=labels, K = K, M = M, Q = Q, iterations = iterations, seed = seed, α = α, η = η)

end


function enslvm_spectra(X, C, W, Z, scales; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

    # sort out dimensions

    D, N = size(X); @assert(size(X) == size(C))

    # report

    @printf("Running enslvm_spectra with K=%d and data of %d number of data items of dimension %d\n", K, N, D)
    @printf("\t number of BLAS threads is %d\n", BLAS.get_num_threads())
    @printf("\t number of julia threads is %d\n", Threads.nthreads())

    # define ensemble output

    t(x) = softmax(x, dims=1)

    f = net(Q = Q, M = M, D = D, o = t)
    
    F(W, z) = mapreduce(w -> f(w, z), +, W) / K
    
    B = Matrix(reduce(hcat, bootstrapweights(K, N; rg = MersenneTwister(seed)))')
    


    #-------------------------------------------
    function pack(Zₙ, sₙ) 
    #-------------------------------------------

        [vec(Zₙ); sqrt(sₙ)]

    end


    #-------------------------------------------
    function unpack(p)
    #------------------------------------------- 

        local Zₙ = @view p[1:end-1]
        
        local sₙ = p[end]^2

        return Zₙ, sₙ

    end


    #-------------------------------------------
    function objective_Zₙ(g, Xₙ, Cₙ, bₙ, p)
    #-------------------------------------------

        local Zₙ, sₙ = unpack(p)

        @assert(length(Zₙ) == Q)

        local aux = zero(eltype(p))

        for k in 1:K

            aux += bₙ[k] * sum(abs2.(Xₙ - sₙ * g[k](Zₙ)) ./ Cₙ)
            
        end

        # penalty on coordinate

        aux += η*sum(abs2.(Zₙ))

        return aux

    end


    #-------------------------------------------
    function objective_single_member(w, Z, s, b)
    #-------------------------------------------

        sum((abs2.(X - f(w, Z)*Diagonal(s)) ./ C)*Diagonal(b)) + α*sum(abs2.(w)) # penalty on weights

    end


    #-------------------------------------------
    function ensemble_performance(W, Z, s)
    #-------------------------------------------

        local aux = 0.0

        for k in 1:K

            aux += sum((abs2.(X - f(W[k], Z)*Diagonal(s)) ./ C)*Diagonal(B[k,:]))

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
                            
                local Xₙ = @view X[:,n]
                
                local Cₙ = @view C[:,n]

                local bₙ = @view B[:,n]

                local obj(x) = objective_Zₙ(g, Xₙ, Cₙ, bₙ, x)

                local grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

                local result = optimize(obj, grad!, pack(Z[:,n], scales[n]), LBFGS(), optZ)
                
                Z[:,n], scales[n] = unpack(result.minimizer)

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

                local b = @view B[k,:]
             
                obj(w) = objective_single_member(w, Z, scales, b)
                
                grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

                W[k] = optimize(obj, grad!, W[k], LBFGS(), optW).minimizer
                
            end

            next!(pr)

        end

        GC.gc(true)

        @printf("Iteration %d, ensemble performance is %f\n", iter, ensemble_performance(W, Z, scales))

    end


    #-------------------------------------------------
    # return results
    #-------------------------------------------------

    function predict(z)

        F(W,z)

    end
    
    predict, W, Z, scales, B
    
end


