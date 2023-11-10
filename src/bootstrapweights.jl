function bootstrapweights(K, N; rg = rg)

    [bootstraphelper(N; rg = rg) for _ in 1:K]

end


function bootstraphelper(N; rg = rg)
    
    w = zeros(N)

    for _ in 1:N

        lucky = ceil(Int, rand(rg)*N)

        w[lucky] += 1

    end

    return w

end