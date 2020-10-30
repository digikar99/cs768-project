# This is a part of GraphAttacks.jl
# Each of these functions return a (dense) score matrix corresponding to
# each node-pair in the graph

function adamic_adar(graph::SimpleGraph)
    g = graph
    adj_sparse = LightGraphs.LinAlg.adjacency_matrix(g)
    inv_log    = begin
        inv_log = zeros(nv(g))
        for i = 1:nv(g)
            num_neigh = length(neighbors(g, i))
            inv_log[i] = (num_neigh < 2 ? 0 : 1/log(num_neigh))
        end
        inv_log
    end

    v_inv_log  = adj_sparse .* inv_log
    # println(size(adj_sparse), size(v_inv_log))
    aa_mat     = adj_sparse * v_inv_log
    aa_mat
end

function katz(graph::SimpleGraph, beta::AbstractFloat)
    g = graph
    adj_sparse = LightGraphs.LinAlg.adjacency_matrix(g)
    katz_mat = inv(I(nv(g)) - beta * Matrix(adj_sparse)) - I(nv(g))
    katz_mat
end

function katz(graph::SimpleGraph, beta::AbstractFloat, u::Int, v::Int, epsilon::Float64=1e-10)
    g = graph
    adj_sparse = LightGraphs.LinAlg.adjacency_matrix(g)

    orig_adj  = float(Matrix(adj_sparse))
    orig_beta = beta
    adj = reshape(orig_adj[u, 1:end], (1,nv(g)))
    katz = 0.0
    while beta > epsilon
        # println(i)
        katz += beta * adj[v]
        adj  *= orig_adj
        beta *= orig_beta
    end
    katz
end


function katz(adj_matrix::Matrix, beta::AbstractFloat, u::Int, v::Int, epsilon::Float64=1e-10)
    # g = graph
    # adj_sparse = LightGraphs.LinAlg.adjacency_matrix(g)

    # orig_adj  = float(Matrix(adj_sparse))
    orig_adj  = adj_matrix
    orig_beta = beta
    adj = reshape(orig_adj[u, 1:end], (1, size(adj_matrix, 1)))
    katz = 0.0
    while beta > epsilon
        # println(i)
        katz += beta * adj[v]
        adj  *= orig_adj
        beta *= orig_beta
    end
    katz
end

# """
#     Assumes that the graph is connected
# """
# function hitting_time(graph::SimpleGraph)
#     adj_sparse = LightGraphs.LinAlg.adjacency_matrix(graph)
#     adj = Matrix(adj_sparse)
#     stochastic_matrix = adj ./ sum(adj, dims=1)
#     eigen_mat = eigvecs(stochastic_matrix)
#     stationary_distribution = eigen_mat[:, end]

# end

function average_commute_time(graph::SimpleGraph)
    num_vertices = nv(graph)
    g = graph

    D = I(num_vertices)*1.0
    for i = 1:num_vertices
        D[i,i] = degree(g, i)
    end

    A = LightGraphs.LinAlg.adjacency_matrix(g)
    L = Matrix(D) .- Matrix(A)

    L_plus = pinv(L)

    S = zeros((num_vertices, num_vertices))
    for i = 1:num_vertices
        for j = 1:num_vertices
            # ignoring constant factor |E|
            # See: https://dl.acm.org/doi/10.1145/3012704
            S[i, j] = L_plus[i,i] + L_plus[j,j] - 2 * L_plus[i,j]
        end
    end
    1 ./ S
end
