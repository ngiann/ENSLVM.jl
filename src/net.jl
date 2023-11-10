struct net 

    Q
    M
    D
    h
    o
end

function net(; Q::Int = Q, M::Int = M, D::Int = D, h = tanh, o = identity)

    net(Q, M, D, h, o)

end

numparam(n::net) = (n.M*n.Q + n.M) + (n.D*n.M + n.D)


function unpack(n::net, param)

    Q, M, D = n.Q, n.M, n.D

    MARK = 0

    b1 = param[MARK+1:MARK+M]

    MARK += M

    W1 = reshape(param[MARK+1:MARK+M*Q], M, Q)

    MARK += M*Q

    b2 = param[MARK+1:MARK+D]

    MARK += D

    W2 = reshape(param[MARK+1:MARK+D*M], D, M)

    MARK += D*M

    @assert(MARK == length(param))

    return b1, W1, b2, W2

end

function (n::net)(param, X)

    b1, W1, b2, W2 = unpack(n, param)

    # W1 is M×Q, X is Q×N

    α = n.h.(W1*X .+ b1) # activations are M×N

    return n.o(W2*α .+ b2)

end

function (n::net)(param)

    b1, W1, b2, W2 = unpack(n, param)

    function pred(X)

        # W1 is M×Q, X is Q×N

        α = n.h.(W1*X .+ b1) # activations are M×N

        return n.o(W2*α .+ b2)

    end

end


function Base.show(io::IO, n::net)
    print(io, "net with ",n.Q ," inputs, ",n.M," hidden units and ",n.D , " outputs.\n Activation function is ",n.h, ". \n Output function is ",n.o)
end