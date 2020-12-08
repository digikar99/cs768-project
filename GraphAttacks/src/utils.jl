
using LinearAlgebra

function argmaximum(f::Function, it)
    max_so_far = typemin(Float64)
    arg        = nothing
    for i in it
        score = f(i)
        if score > max_so_far
            max_so_far = score
            arg        = i
        end
    end
    arg
end


function sum_of_powers(x, power)
    """For each x_i, computes sum_{r=1}^{pow) x_i^r (elementwise sum of powers).
    :param x: shape [?]
        Any vector
    :param pow: int
        The largest power to consider
    :return: shape [?]
        Vector where each element is the sum of powers from 1 to pow.
    """
    n = size(x,1)
    sum_powers = zeros(power, n)
    for i in 1:power
        sum_powers[i] = x.^i
    end

    return sum(sum_powers,dims=1)
end


function sum_of_powers_of_transition_matrix(adj_matrix, pow)
    """Computes sum_{r=1}^{pow) (D^{-1}A)^r.
    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param pow: int
        Power exponent
    :return: sp.csr_matrix
        Sum of powers of the transition matrix of a graph.
    """
    deg = sum(adj_matrix,dims=2) #tested
    deg=map(x->max(1,x),deg)#chitr
    transition_matrix = Diagonal(1 ./ deg) * adj_matrix

    sum_of_powers = transition_matrix
    last_ = transition_matrix
    for i in 1:pow
        last_ = last_*transition_matrix
        sum_of_powers += last_
    end

    return sum_of_powers
end

