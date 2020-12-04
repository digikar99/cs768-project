
module SRW

using Distances
using LightGraphs

# The following code is based on: https://arxiv.org/abs/1011.4071 ##############

function principle_eigenvector(matrix, eps=1e-6)
    # Power iteration to find principle eigenvector:
    # https://en.wikipedia.org/wiki/Power_iteration
    prev = rand(size(matrix,1))
    next = rand(size(matrix,1))
    matrixT = transpose(matrix)
    while maximum(abs.(next-prev))>eps
        prev = next
        next = matrixT * prev
        next ./= norm(next)
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
end

function make_SRW(graph, node_features;
                  restart_probability=0.1, weights=nothing)
    num_nodes, num_features = size(node_features)
    # TODO: Handle the case wherein edge_strength_function is not a dot product
    num_weights = num_features
    if weights == nothing weights = zeros(num_weights) end
    
    SupervisedRandomWalker(
        weights,
        restart_probability,
        graph,
        node_features,
        zeros(num_nodes, num_nodes),
        zeros(num_nodes, num_nodes),
        zeros(num_nodes, num_nodes),
        zeros(num_nodes, num_nodes),
        zeros(num_nodes, num_nodes, num_weights)
    )
end

function edge_features(srw::SupervisedRandomWalker, u, v)
    @. abs(srw.node_features[u] - srw.node_features[v])
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
            d      = node_features[v,:]
            psi_uv = edge_features(u, d)
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
        Q_tmp[:, u] .+= restart_probability
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

function loss(srw::SupervisedRandomWalker,
              loss_fn::Function = hinge_loss)

    graph             = srw.graph
    stationary_scores = srw.stationary_scores
    num_nodes         = nv(graph)

    total_loss = 0
    for s = 1:num_nodes
        for d in neighbors(graph, s)
            for l = 1:num_nodes
                if not(has_edge(s, l))
                    pd = stationary_scores[s,d]
                    pl = stationary_scores[s,l]
                    total_loss += loss_fn(pd, pl)
                end
            end
        end
    end

    return total_loss, total_loss + norm2(srw.weights)
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
    edge_strength_derivatives = zeros(num_nodes, num_weights)

    # for w = 1:num_weights
    for j = 1:num_nodes
        for u = 1:num_nodes
            edge_strength_derivatives[u, :] .= edge_strength_function_derivative(
                weights,
                edge_features(node_features[j], node_features[u])
            )
        end

        # sum along nodes
        num1 = sum(edge_strengths[j, :])              # number
        num2 = sum(edge_strength_derivatives, dims=1) # vector of length num_weights
        sum_num2 = sum(num2)
        
        for u = 1:num_nodes
            if has_edge(g, start_node, u)
                @. dQ_by_dw[:, j, u] = (1-alpha) \
                    *(edge_strength_derivatives[u, :] * num1 \
                      - edge_strengths[j,u] * num2) \
                      /(num1^2)
            else
                dQ_by_dw[:, j, u] .= 0
            end
        end
    end
    # end
end

function dp_by_dw(srw::SupervisedRandomWalker, start_node, target_node, eps = 1e-3)
    # This function will be called O(|V|^3) times from inside grad_term_2
    # for every gradient calculation.
    
    num_nodes   = nv(srw.graphs)
    num_weights = size(srw.weights, 1)
    derivatives = zeros(num_weights)
    new_derivatives = zeros(num_weights)
    alpha       = srw.restart_probability
    Q_tmp       = srw.Q_tmp
    p           = srw.stationary_scores[start_node, :]
    u           = target_nodes
    # srw.dQ_by_dw already corresponds to the start node by cache_dQ_by_dw
    dQ_by_dw    = srw.dQ_by_dw[:, u, :] 

    # TODO: Optimize further by caching values into a matrix
    # Q_tmp[:, start_node] .+= alpha
    for k = 1:num_weights
        while abs.(new_derivatives[k] - derivatives[k]) > eps
            derivatives[k] = new_derivatives[k]
            new_derivatives[k] = sum(
                # Each of these should be a vector of length num_nodes
                @. (Q_tmp[:, start_node] + alpha) * new_derivatives \
                + p * dQ_by_dw[:, k]
            )
        end
    end
    # Q_tmp[:, start_node] .-= alpha
    
    new_derivatives # should be a vector of length num_weights
end

function grad_term_2(srw::SupervisedRandomWalker, loss_fn_grad=hinge_loss_grad)
    # The second main term of the gradient summing over s and l,d

    num_weights       = size(srw.weights, 1)
    g                 = srw.graph
    stationary_scores = srw.stationary_scores
    # TODO: Speed up by avoiding allocation of grad; refer back to function 'fit'
    grad = zeros(num_weights)
    for s = 1:num_nodes
        # TODO: Handle each weight separately to save some space
        cache_dQ_by_dw(srw, s)
        for d in neighbors(g, s)
            for l = 1:num_nodes
                if not(has_edge(g, s, l))
                    pd = stationary_scores[s,d]
                    pl = stationary_scores[s,l]
                    d_pd = dp_by_dw(srw, s, d)
                    d_pl = dp_by_dw(srw, s, l)
                    # weights and therefore, grad, and therefore, the return
                    # value of dp_by_dw should be a vector of length num_weights
                    @. grad += loss_fn_grad(pd, pl) * (d_pl - d_pd)
                end
            end
        end
    end
    grad # should be a vector of length num_weights
end

# TODO: Add option to avoid regularization
function fit(srw::SupervisedRandomWalker, lambda, target_loss=1e-3, learning_rate=0.1)
    graph         = srw.graph
    node_features = srw.node_features
    weights       = srw.weights
    regularized_loss_score = loss(srw)[2]
    
    while regularized_loss_score > target_loss
        predict(srw)        
        # weights (and therefore, grad) should be a vector of length num_weights
        # and therefore, also the value returned by grad_term_2
        grad = 2*weights + grad_term_2(srw)
        weights .-= learning_rate*grad
        regularized_loss_score = loss(srw)[2]
    end
end

end # end SRW module
