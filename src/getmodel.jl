function getmodel(W::Vector{Vector{T}}; Q = Q, D = D) where T

    K = length(W)

    M = round(Int, (length(W[1]))/(Q + 1 + D))

    @printf("Instantiating network ensemble with %d members with M=%d, input is %d, output is %d\n", K, M, Q, D)

    t(x) = softmax(x, dims=1)

    f = net(Q = Q, M = M, D = D, o = t)

    g = map(f, W)

    F(z) = mapreduce(gᵢ -> gᵢ(z), +, g) / K
    

    return F

end

function getmodel(filename::String; Q = Q, D = D)

    W = JLD2.load(filename, "W")

    getmodel(W; Q = Q, D = D)

end