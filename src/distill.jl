function distill(X, C, W, Z, scales; iterations = 1, α = 0.1)

    # sort out dimensions

    D, N = size(X); @assert(size(X) == size(C))

    K = length(W)

    Q = size(Z, 1)
    
    M = round(Int, (length(W[1])-D)/(Q + 1 + D))

    # report

    @printf("Running distill with K=%d, M = %d and data of %d number of data items of dimension %d\n", K, M, N, D)
    @printf("\t number of BLAS threads is %d\n", BLAS.get_num_threads())
    @printf("\t number of julia threads is %d\n", Threads.nthreads())

    # define ensemble output

    t(x) = softmax(x, dims=1)

    f = net(Q = Q, M = M, D = D, o = t)
    
    
    # calculate targets

    T = let

        F(W, z) = mapreduce(w -> f(w, z), +, W) / K
    
        F(W, Z)*Diagonal(scales)
        
    end
    

    #-------------------------------------------
    function objective(w)
    #-------------------------------------------

       sum((abs2.(T - f(w, Z)*Diagonal(scales)) ./ C)) + α*sum(abs2.(w))

    end


    opt = Optim.Options(iterations = iterations, show_trace = true, show_every=1)
    
    grad!(s, x) = copyto!(s, Zygote.gradient(objective, x)[1])

    wdistill = optimize(objective, grad!, randn(numparam(f)), LBFGS(), opt).minimizer
                
    
    f(wdistill), objective, wdistill
    
end