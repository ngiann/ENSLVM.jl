struct net{F}
    Q
    M
    D
    o::F
end

function net(; Q::Int = Q, M::Int = M, D::Int = D, o = NNlib.softmax)

    net(Q, M, D, o)

end

numparam(n::net) = (n.M*n.Q + n.M) + (n.D*n.M + n.D)


function unpack(n::net, param)

    Q, M, D = n.Q, n.M, n.D

    MARK = 0

    b1 = @view param[MARK+1:MARK+M]

    MARK += M

    W1_ = @view param[MARK+1:MARK+M*Q]
    
    W1 = reshape(W1_, M, Q)

    MARK += M*Q

    b2 = @view param[MARK+1:MARK+D]

    MARK += D

    W2_ = @view param[MARK+1:MARK+D*M]
    
    W2 = reshape(W2_, D, M)

    MARK += D*M

    @assert(MARK == length(param))

    return b1, W1, b2, W2

end

function (n::net)(param, X)

    b1, W1, b2, W2 = unpack(n, param)

    # W1 is M×Q, X is Q×N

    α = tanh.(W1*X .+ b1) # activations are M×N

    return n.o(W2*α .+ b2)

end

function (n::net)(param)

    b1, W1, b2, W2 = unpack(n, param)

    function pred(X)

        # W1 is M×Q, X is Q×N

        α = tanh.(W1*X .+ b1) # activations are M×N

        return n.o(W2*α .+ b2)

    end

end


function Base.show(io::IO, n::net)
    print(io, "net with ",n.Q ," inputs, ",n.M," hidden units and ",n.D , " outputs.\n Activation function is ",n.h, ". \n Output function is ",n.o)
end
