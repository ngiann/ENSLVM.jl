function enslvm(X, labels = nothing; K = 10, M = 10, Q = 2, iterations = 1, seed = 1, plot_every = 10, α = 0.1, η = 1.0, t = identity)

    rg = MersenneTwister(seed)

    D, N = size(X)

    # report

    @printf("Running ENSLVM with K=%d and data of %d number of data items of dimension %d\n", K, N, D)


    # define ensemnle output

    f = net(Q = Q, M = M, D = D, o = t)
    
    F(W, Z) = mapreduce(w->f(w, Z), +, W) / K
    

    # randomly initialise network parameters
    
    W = [randn(numparam(f)) for _ in 1:K]


    # randomly initialise latent

    Z = randn(Q, N)


    # draw bootstrap weights
    
    B = bootstrapweights(K, N; rg = rg)


    #-------------------------------------------
    function objective_Z(W, Z)
    #-------------------------------------------

        @assert(size(Z) == (Q, N))

        local aux = zero(eltype(Z))

        for k in 1:K

            aux += dot(B[k], sum(abs2.(X - f(W[k], Z)), dims=1))

        end

        # penalty on coordinates

        aux += η*sum(abs2.(Z))

        aux

    end


    #-------------------------------------------
    function objective_single_member(w, Bₖ, Z)
    #-------------------------------------------

        @assert(size(Z) == (Q, N))

        @assert(length(Bₖ) == N)

        dot(Bₖ, sum(abs2.(X - f(w, Z)), dims=1)) + α*sum(abs2.(w))

    end


    for iter in 1:iterations

        #-----------------------
        # hold ensemble fixed
        #-----------------------

        let

            opt = Optim.Options(iterations = 3, show_trace = false, show_every=1)

            obj(zvec) = objective_Z(W, reshape(zvec, Q, N))

            grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

            result = optimize(obj, grad!, vec(Z), LBFGS(), opt)
            
            Z = reshape(result.minimizer, Q, N)

        end


        #-----------------------
        # plot latent
        #-----------------------

        if mod(iter, plot_every)==1 && (Q == 2 || Q == 3)

            figure(0)
            cla()
            if isnothing(labels)
                if Q == 2
                    plot(Z[1,:], Z[2,:], "o")
                else
                    plot3D(Z[1,:], Z[2,:],Z[:,3], "o")
                end
            
            else
                for l in unique(labels)
                    idx = findall(labels .== l)
                   if Q == 2
                     plot(Z[1,idx], Z[2,idx], "o")
                   else
                    plot3D(Z[1,idx], Z[2,idx],Z[3, idx], "o")
                   end
                end
            end

            axis("equal")
        end


        #-----------------------
        # hold latent fixed
        #-----------------------

        let

            opt = Optim.Options(iterations = 3, show_trace = false, show_every=1)

            for k in 1:K

                let

                    obj(w) =  objective_single_member(w, B[k], Z)
                    
                    grad!(s, x) = copyto!(s, Zygote.gradient(obj, x)[1])

                    W[k] = optimize(obj, grad!, W[k], LBFGS(), opt).minimizer
                   
                end

            end

        end

        @printf("Iteration %d\n", iter)

    end

    z -> F(W, z)
    
end


