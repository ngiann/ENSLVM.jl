struct net{F}
    Q
    M
    D
    o::F
end

function net(; Q::Int = Q, M::Int = M, D::Int = D, o = NNlib.softmax)

    net(Q, M, D, o)

end

numparam(n::net) = (n.M*n.Q + n.M) + (n.D*n.M)


function unpack(n::net, param)

    Q, M, D = n.Q, n.M, n.D

    MARK = 0

    b1 = @view param[MARK+1:MARK+M]

    MARK += M

    W1_ = @view param[MARK+1:MARK+M*Q]
    
    W1 = reshape(W1_, M, Q)

    MARK += M*Q

    W2_ = @view param[MARK+1:MARK+D*M]
    
    W2 = reshape(W2_, D, M)

    MARK += D*M

    @assert(MARK == length(param))

    return b1, W1, W2

end

function (n::net)(param, X)

    b1, W1, W2 = unpack(n, param)

    α = tanh.(W1*X .+ b1) # W1 is M×Q, X is Q×N

    return n.o(W2*α)      # activations are M×N

end

function (n::net)(param)

    b1, W1, W2 = unpack(n, param)

    function pred(X)

        α = tanh.(W1*X .+ b1)  # W1 is M×Q, X is Q×N

        return n.o(W2*α)       # activations are M×N

    end

end


function Base.show(io::IO, n::net)
    print(io, "net with ",n.Q ," inputs, ",n.M," hidden units and ",n.D , " outputs.\n Number of weights is ",numparam(n),". \n")
end
