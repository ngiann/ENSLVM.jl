#-------------------------------------------
function ensemblehessian(X, C, B, W, Z, s;  K = 10, M = 10, Q = 2)
#-------------------------------------------

    @assert(length(B) == length(W))

    D, N = size(X); @assert(size(X) == size(C))

    t(x) = softmax(x, dims=1)

    f = net(Q = Q, M = M, D = D, o = t)


    local ℋ = zeros(numparam(f), numparam(f))#[zeros(numparam(f), numparam(f)) for _ in 1:Threads.nthreads()]



    function negll(wₖ, k)

        local ℓ = zero(eltype(wₖ))
        
        local gₖ = f(wₖ)
        
        for n in 1:1

            local diff = X[:,n] - s[n] * gₖ(Z[:,n]) 
            
            ℓ += 0.5 * sum((abs2.(diff) ./ C[:,n]))*B[k][n] # check if there should be a minus sign in front

        end

        return ℓ

    end


    pr = Progress(K)
    
    @showprogress for k in 1:K

        ℋ += ForwardDiff.hessian(w -> negll(w, k), W[k])
    
        next!(pr)

    end
    

    return ℋ

end
