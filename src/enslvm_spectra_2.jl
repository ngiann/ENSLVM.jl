function enslvm_spectra_2(X, C; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

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

    scales = ones(K, N)


    # call model with initialised parameters

    enslvm_spectra_2(X, C, W, Z, scales; labels=labels, K = K, M = M, Q = Q, iterations = iterations, seed = seed, α = α, η = η)

end


function enslvm_spectra_2(X, C, W, Z, scales; labels=nothing, K = 10, M = 10, Q = 2, iterations = 1, seed = 1, α = 0.1, η = 1.0)

    #----------------------
    # sort out dimensions
    #----------------------

    D, N = size(X); @assert(size(X) == size(C))


    #----------------------
    # report
    #----------------------

    @printf("Running enslvm_spectra_2 with K=%d and data of %d number of data items of dimension %d\n", K, N, D)
    @printf("\t number of BLAS threads is %d\n", BLAS.get_num_threads())
    @printf("\t number of julia threads is %d\n", Threads.nthreads())


    #----------------------
    # define ensemble output
    #----------------------

    t(x) = softmax(x, dims=1)

    f = net(Q = Q, M = M, D = D, o = t)
    
    F(W, z) = mapreduce(w -> f(w, z), +, W) / K
    

    #----------------------
    # Bootstrap weights
    #----------------------

    B = Matrix(reduce(hcat, bootstrapweights(K, N; rg = MersenneTwister(seed)))')
    


    #-------------------------------------------
    function packZs(Zₙ, sₙ) 
    #-------------------------------------------

        [vec(Zₙ); sqrt.(sₙ)]

    end


    #-------------------------------------------
    function unpackZs(p)
    #-------------------------------------------
    
        @assert(length(p) == Q + K)

        local Zₙ = @view p[1:Q]
        
        local sₙ = p[Q+1:end].^2

        return Zₙ, sₙ

    end


    #-------------------------------------------
    function packWs(wₖ, sₖ) 
    #-------------------------------------------
    
        [vec(wₖ); sqrt.(sₖ)]
    
    end

    #-------------------------------------------
    function unpackWs(p)
    #------------------------------------------- 
    
        @assert(length(p) == numparam(f) + N)

        local wₖ = p[1:numparam(f)]
        
        local sₖ = p[numparam(f)+1:end].^2

        return wₖ, sₖ
    
    end

    

    #-------------------------------------------
    function objective_Zₙ(g, Xₙ, Cₙ, bₙ, p)
    #-------------------------------------------

        local Zₙ, sₙ = unpackZs(p)

        @assert(length(Zₙ) == Q); @assert(length(sₙ) == K)

        local aux = zero(eltype(p))

        for k in 1:K

            if bₙ[k] == 0
                continue
            end

            aux += bₙ[k] * sum(abs2.(Xₙ - sₙ[k] * g[k](Zₙ)) ./ Cₙ)
            
        end

        # penalty on coordinate

        aux += η*sum(abs2.(Zₙ))

        return aux

    end


    #-------------------------------------------
    function objective_single_member(p, Z, b)
    #-------------------------------------------

        @assert(length(b) == N)
        
        local wₖ, sₖ = unpackWs(p)

        sum((abs2.(X - f(wₖ, Z)*Diagonal(sₖ)) ./ C)*Diagonal(b)) + α*sum(abs2.(wₖ)) # penalty on weights

    end


    #-------------------------------------------
    function ensemble_performance(W, Z, scales)
    #-------------------------------------------

        @assert(length(W) == K); @assert(size(Z) == (Q, N)); @assert(size(scales) == (K, N))

        local aux = 0.0

        for k in 1:K

            aux += sum((abs2.(X - f(W[k], Z)*Diagonal(scales[k,:])) ./ C)*Diagonal(B[k,:]))

        end

        return aux / K

    end




    for iter in 1:iterations

        #----------------------------------------------------
        # hold ensemble fixed, optimise latent coordinates Z
        #----------------------------------------------------
        
        let 
          
            pr = Progress(N, desc = "Latent coordinates")
          
            optZ = Optim.Options(iterations = min(iter, 15), show_trace = false, show_every=1)
       
            local g = map(f, W)

           Threads.@threads for n in 1:N
                            
                local Xₙ = @view X[:,n]
                
                local Cₙ = @view C[:,n]

                local bₙ = @view B[:,n]

                local obj(x) = objective_Zₙ(g, Xₙ, Cₙ, bₙ, x)

                local grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

                local result = optimize(obj, grad!, packZs(Z[:,n], scales[:,n]), LBFGS(), optZ)
                
                Z[:,n], scales[:,n] = unpackZs(result.minimizer)

                next!(pr)

            end

        end


        
        #-------------------------------------------------
        # hold latent fixed, optimise ensemble weights wₖ
        #-------------------------------------------------

        pr = Progress(K, desc = "Ensemble members")

        optW = Optim.Options(iterations = min(iter, 15), show_trace = false, show_every=1)
    
        Threads.@threads for k in 1:K

            let

                local b = @view B[k,:]
             
                obj(p) = objective_single_member(p, Z, b)
                
                grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

                local result = optimize(obj, grad!, packWs(W[k],scales[k,:]), LBFGS(), optW)
                
                W[k], scales[k,:] = unpackWs(result.minimizer)
            end

            next!(pr)

        end

        @printf("Iteration %d, ensemble performance is %f\n", iter, ensemble_performance(W, Z, scales))
        
        GC.gc(true)

    end


    #-------------------------------------------------
    # return results
    #-------------------------------------------------

    predict(z) = F(W,z)
    
    reconstruct(i, k) = scales[k,i] * f(W[k], Z[:,i])

    function reconstruct(i)

        aux = zeros(D)
        
        count = 0.0

        for k in 1:K
            
            if B[k, i] == 0
                continue
            else
                count += 1
                aux += reconstruct(i, k)
            end

        end

        return aux / count

    end

    
    predict, reconstruct, W, Z, scales, B
    
end


