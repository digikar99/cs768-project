function greedy_katz(graph::SimpleGraph, target::SimpleGraph, budget::Int,
                     katz_beta::AbstractFloat)
    # Incomplete
    # Delete those edges that result in maximum decrease of the Katz score of the target links
    # base_katz_mat = katz(graph, katz_beta)
    base_adj_mat = float(Matrix(LightGraphs.LinAlg.adjacency_matrix(graph)))

    function target_sum_scorer(adj_mat::Matrix)
        total = 0.0
        for e in edges(target)
            u = src(e)
            v = dst(e)
            # total += score_matrix[u,v]
            total += katz(adj_mat, katz_beta, u, v)
        end
        total
    end

    base_katz_score = target_sum_scorer(base_adj_mat)
    best_diff       = 0.0
    opt_g       = copy(graph)
    opt_adj_mat = copy(base_adj_mat)

    for b = 1:budget
        num_edges_tried = 0
        new_opt_g       = opt_g
        new_opt_adj_mat = opt_adj_mat
        opt_u = nothing
        opt_v = nothing
        for e in edges(opt_g)
            num_edges_tried += 1
            if num_edges_tried % 100 == 0 println(num_edges_tried) end
            temp = copy(opt_g)
            rem_edge!(temp, e)
            new_opt_adj_mat[src(e), dst(e)] = 0
            new_opt_adj_mat[dst(e), src(e)] = 0
            katz_score = target_sum_scorer(new_opt_adj_mat)
            diff       = base_katz_score - katz_score
            if diff > best_diff
                best_diff = diff
                new_opt_g = temp
                opt_u     = src(e)
                opt_v     = dst(e)
            end
            new_opt_adj_mat[src(e), dst(e)] = 1
            new_opt_adj_mat[dst(e), src(e)] = 1
        end
        opt_g = new_opt_g
        new_opt_adj_mat[opt_u, opt_v] = 0
        new_opt_adj_mat[opt_v, opt_u] = 0
    end
    opt_g
end

function greedy_cnd(graph::SimpleGraph, target::SimpleGraph, budget::Int)
    # https://arxiv.org/abs/1809.08368
    g     = graph
    new_g = copy(g)
    for e in edges(target)
        u = src(e)
        v = dst(e)
        # new_g since common neighbors may change with updates
        CNs = sort!(common_neighbors(new_g, u, v), rev=true)
        for opt_w in CNs
            if has_edge(g, u, opt_w)
                rem_edge!(new_g, u, opt_w)
            elseif has_edge(g, v, opt_w)
                rem_edge!(new_g, v, opt_w)
            else
                error("We weren't supposed to arrive here!")
            end
        end
    end
    new_g
end

