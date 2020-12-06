
module SRW

using Distances
using LinearAlgebra
using LightGraphs
using Random

# The following code is based on: https://arxiv.org/abs/1011.4071 ##############

MAX_ITERATIONS = 10000

function principle_eigenvector(matrix, eps=1e-6)
    # eigen(matrix).vectors[:, end]
    # Power iteration to find principle eigenvector:
    # https://en.wikipedia.org/wiki/Power_iteration
    prev = rand(size(matrix,1))
    next = rand(size(matrix,1))
    iter = 0
    # matrixT = transpose(matrix)
    while maximum(abs.(next-prev))>eps && iter < MAX_ITERATIONS
        prev .= next
        next = matrix * prev
        next ./= norm(next)
        iter += 1
        # println(prev, " ", next, " ", maximum(abs.(next-prev)))
    end
    return next
end

# ------------------------------------------------------------------------------

## File Structure: main parts
#  - SupervisedRandomWalker and make_SRW
#  - predict
#  - loss
#  - fit
##

mutable struct SupervisedRandomWalker
    weights
    restart_probability # alpha
    graph
    node_features

    # For optimization purposes
    edge_strengths
    unrestarted_scores
    stationary_scores

    # Q_tmp should be (1-alpha)*unrestarted_scores
    Q_tmp # avoid allocation wherever possible - especially inside dp_by_dw
    dQ_by_dw
    dp_by_dw
    new_dp_by_dw

    RAND_V_BY_W_1
    RAND_V_BY_W_2
end

function make_SRW(graph, node_features;
                  restart_probability=0.1, weights=nothing)
    node_features = node_features[1:nv(graph), :]
    num_nodes, num_features = size(node_features)
    # TODO: Handle the case wherein edge_strength_function is not a dot product
    num_weights = 2*num_features
    if weights == nothing weights = rand(num_weights) end

    # for v=1:num_nodes # normalize
        # node_features[v,:] ./= sum(node_features[v,:])
    # end

    SupervisedRandomWalker(
        weights,
        restart_probability,
        graph,
        node_features,

        zeros(num_nodes, num_nodes),
        zeros(num_nodes, num_nodes),
        zeros(num_nodes, num_nodes),

        zeros(num_nodes, num_nodes),
        zeros(num_nodes, num_nodes, num_weights),
        rand(num_nodes, num_weights),
        rand(num_nodes, num_weights),

        rand(num_nodes, num_weights),
        rand(num_nodes, num_weights)
    )
end

function reset(srw::SupervisedRandomWalker)
    num_nodes, num_features = size(srw.node_features)
    num_weights  = size(srw.weights,1)
    srw.weights  = rand(num_weights)
    srw.dp_by_dw = rand(num_nodes, num_weights)
    srw.new_dp_by_dw = rand(num_nodes, num_weights)
    nothing
end

function edge_features(srw::SupervisedRandomWalker, u, v)
    # srw.node_features[u,:] .* srw.node_features[v,:]
    vcat(srw.node_features[u,:], srw.node_features[v,:])
end

"The f_w(psi_uv) of the paper"
function edge_strength_function(w, psi_uv) dot(w, psi_uv) end
function df_by_dw(w, psi_uv) psi_uv end
edge_strength_function_derivative = df_by_dw

# ------------------------------------------------------------------------------

"""
node_features -> Each row corresponds to the features of a node,
                 and thus is a num_nodes X num_features matrix

"""

function predict(srw::SupervisedRandomWalker)
    # Predicts for all node pairs
    weights                 = srw.weights
    restart_probability     = srw.restart_probability
    node_features           = srw.node_features
    num_nodes, num_features = size(node_features)
    unrestarted_scores      = srw.unrestarted_scores # The Q' of the paper
    stationary_scores       = srw.stationary_scores  # The p of the paper
    edge_strengths          = srw.edge_strengths     # a_uv of the paper

    function score_node_neighbors(node)
        u = node
        for v = 1:num_nodes
            psi_uv = edge_features(srw, u, v)
            a_uv   = edge_strength_function(weights, psi_uv)
            edge_strengths[u,v] = a_uv
            unrestarted_scores[u,v] = a_uv
        end
        unrestarted_scores[u, :] ./= sum(unrestarted_scores[u, :])
    end

    for u = 1:num_nodes
        score_node_neighbors(u)
    end


    # avoid allocation of Q
    Q_tmp = srw.Q_tmp
    Q_tmp .= (1-restart_probability).*unrestarted_scores
    for u = 1:num_nodes
    # for u = 1:2
        Q_tmp[:, u] .+= restart_probability
        # print(Q_tmp)
        stationary_scores[u, :] .= principle_eigenvector(Q_tmp)
        Q_tmp[:, u] .-= restart_probability
    end

    return stationary_scores
end

# ------------------------------------------------------------------------------

function hinge_loss(larger, smaller)
    if larger >= smaller
        0
    else
        smaller - larger
    end
end

function hinge_loss_grad(larger, smaller) smaller>larger end

"""
Returns two values: unregularized loss, regularized loss
"""

# TODO: Incorporate lambda
function loss(srw::SupervisedRandomWalker,
              loss_fn::Function = hinge_loss)

    graph             = srw.graph
    stationary_scores = srw.stationary_scores
    num_nodes         = nv(graph)

    total_loss = 0
    for s = 1:num_nodes
        for d in neighbors(graph, s)
            for l = 1:num_nodes
                if !(has_edge(graph, s, l))
                    pd = stationary_scores[s,d]
                    pl = stationary_scores[s,l]
                    total_loss += loss_fn(pd, pl)
                end
            end
        end
    end

    return total_loss, total_loss + norm(srw.weights, 2)
end

# ------------------------------------------------------------------------------

"This corresponds to the expression given between 'Algorithm 1' and 'Final remarks'."
function cache_dQ_by_dw(srw::SupervisedRandomWalker, start_node)

    num_nodes, num_features = size(srw.node_features)
    weights                 = srw.weights
    num_weights             = size(weights, 1)
    dQ_by_dw                = srw.dQ_by_dw
    alpha                   = srw.restart_probability
    g                       = srw.graph
    edge_strengths          = srw.edge_strengths
    node_features           = srw.node_features
    edge_strength_derivatives = zeros(num_nodes, num_weights) # corresponds to a single j

    # for w = 1:num_weights
    for j = 1:num_nodes
        for u = 1:num_nodes
            edge_strength_derivatives[u, :] .= edge_strength_function_derivative(
                weights,
                edge_features(srw, j, u)
            )
        end

        # sum along nodes
        num1 = sum(edge_strengths[j, :])              # number
        # vector of length num_weights
        num2 = reshape(sum(edge_strength_derivatives, dims=1), (num_weights))
        sum_num2 = sum(num2)

        for u = 1:num_nodes
            if has_edge(g, start_node, u)
                # println(map(size, (dQ_by_dw[j,u,:],
                #                    edge_strength_derivatives[u,:],
                #                    num2,
                #                    num1,
                #                    edge_strengths[j,u])))
                @. dQ_by_dw[j, u, :] = (1-alpha) *
                    (edge_strength_derivatives[u, :] * num1 -
                     edge_strengths[j,u] * num2) /
                     (num1^2)
            else
                dQ_by_dw[j, u, :] .= 0
            end
        end
    end
    # end
end

function cache_dp_by_dw(srw::SupervisedRandomWalker, start_node, eps=1e-3)
    num_nodes   = nv(srw.graph)
    num_weights = size(srw.weights, 1)
    dp_by_dw    = srw.dp_by_dw
    new_dp_by_dw = srw.new_dp_by_dw
    alpha       = srw.restart_probability
    Q_tmp       = srw.Q_tmp
    p           = srw.stationary_scores[start_node, :]
    # srw.dQ_by_dw already corresponds to the start node by cache_dQ_by_dw

    # TODO: Are these necessary? These seem to cause a large amount of allocation
    # rand!(dp_by_dw)
    dp_by_dw .= srw.RAND_V_BY_W_1
    # new_dp_by_dw .= srw.RAND_V_BY_W_2

    # Q_tmp[:, start_node] .+= alpha
    i = 0
    for k = 1:num_weights
        while maximum(abs.(new_dp_by_dw[:,k] - dp_by_dw[:,k])) > eps && i<MAX_ITERATIONS
            i += 1
            # println()
            # println("dp_by_dw: ", dp_by_dw[1:10,k])
            # println("new_dp_by_dw: ", new_dp_by_dw[1:10,k])
            dp_by_dw[:,k] .= new_dp_by_dw[:,k]
            for u = 1:num_nodes
                dQ_by_dw    = srw.dQ_by_dw[:, u, :]
                # println("shapes: ", map(size, (Q_tmp[:,start_node],
                #                                alpha,
                #                                new_dp_by_dw[:,k],
                #                                p, dQ_by_dw[:,k])))
                new_dp_by_dw[u,k] = sum(
                    # Each of these should be a vector of length num_nodes
                    @. (Q_tmp[:, start_node] + alpha) * new_dp_by_dw[:,k] +
                    p * dQ_by_dw[:, k]
                )
            end
            # diverges without this
            dp_by_dw[:,k] ./= sum(dp_by_dw[:,k])
            new_dp_by_dw[:,k] ./= sum(new_dp_by_dw[:,k])
        end
    end

    # Q_tmp[:, start_node] .-= alpha
end

function grad_term_2(srw::SupervisedRandomWalker, loss_fn_grad=hinge_loss_grad)
    # The second main term of the gradient summing over s and l,d

    num_weights       = size(srw.weights, 1)
    g                 = srw.graph
    stationary_scores = srw.stationary_scores
    num_nodes         = nv(srw.graph)
    # TODO: Speed up by avoiding allocation of grad; refer back to function 'fit'
    grad              = zeros(num_weights)
    dp_by_dw          = srw.dp_by_dw

    print("  Calculating gradient's second term")
    for s = 1:num_nodes
        print(s, " ")
        # TODO: Handle each weight separately to save some space
        cache_dQ_by_dw(srw, s)
        cache_dp_by_dw(srw, s)
        # println(dp_by_dw)
        for d in neighbors(g, s)
            for l = d+1:num_nodes
                if !(has_edge(g, s, l))
                    # println("    NonEdge: ", (s,l))
                    pd = stationary_scores[s,d]
                    pl = stationary_scores[s,l]
                    d_pd = dp_by_dw[d,:]
                    d_pl = dp_by_dw[l,:]
                    # weights and therefore, grad, and therefore, the return
                    # value of dp_by_dw should be a vector of length num_weights
                    # println(" d_pl, d_pd ", (d_pl, d_pd))
                    @. grad += loss_fn_grad(pd, pl) * (d_pl - d_pd)
                end
            end
        end
    end

    println("\n  Calculated! Sum: ", sum(grad))
    grad # should be a vector of length num_weights
end

# TODO: Add option to avoid regularization
function fit(srw::SupervisedRandomWalker; target_loss=1e-3,
             learning_rate=0.1, num_epochs=10)
    graph         = srw.graph
    node_features = srw.node_features
    weights       = srw.weights
    regularized_loss_score = loss(srw)[2]
    epoch_completed = 0

    while regularized_loss_score > target_loss && epoch_completed < num_epochs
        predict(srw)
        println("Epoch $epoch_completed Loss: $regularized_loss_score")
        # weights (and therefore, grad) should be a vector of length num_weights
        # and therefore, also the value returned by grad_term_2
        grad = 2*weights + grad_term_2(srw)
        weights .-= learning_rate*grad
        regularized_loss_score = loss(srw)[2]
        epoch_completed += 1
    end
    predict(srw)
    println("Epoch $epoch_completed Loss: $regularized_loss_score")
end

end # end SRW module


# fcon = "/home/shubhamkar/ram-disk/citeseer/citeseer.content"
# fcites = "/home/shubhamkar/ram-disk/citeseer/citeseer.cites"
# mg, d = cs.read_into_graph(fcites)
# lg = cs.extract_largest_connected_component(mg, true)
# tg = cs.trim(lg, 5)
# features = cs.read_features(fcon, tg, 100)
# srw = SRW.make_SRW(tg, features)
# SRW.predict(srw)
# SRW.loss(srw)
# SRW.fit(srw)
