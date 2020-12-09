# This file is a part of GraphAttacks.jl

using LightGraphs
# include("utils.jl")

function closed_triad_removal(train_graph::SimpleGraph,
                              test_graph::SimpleGraph,
                              budgets)

    iter=1
    Channel() do channel

        if budgets[iter] == 0
            put!(channel,SimpleGraph(train_graph))
            iter += 1
        end

        for b in 1:maximum(budgets)

            scores = begin
                d = Dict()
                for e in edges(train_graph)
                    d[sort!([src(e), dst(e)])] = 0 # assumes src <= dst
                end
                d
            end

            edge_with_max_score = begin
                for e in edges(test_graph)
                    u = src(e)
                    v = dst(e)
                    # TODO: This allocates memory
                    for n in union(neighbors(train_graph, u), neighbors(train_graph, v))
                        if has_edge(train_graph, u, n) && has_edge(train_graph, v, n)
                            scores[sort!([u,n])] += 1
                            scores[sort!([v,n])] += 1
                        end
                    end
                end
                argmaximum(e->scores[e], keys(scores))
            end
            u, v = edge_with_max_score
            rem_edge!(train_graph, u, v)
            if iter<=length(budgets) && b==convert(Int,budgets[iter])
                put!(channel,SimpleGraph(train_graph))
                iter+=1
            end
        end
    end
end

