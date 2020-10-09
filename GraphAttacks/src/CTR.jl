# This file is a part of GraphAttacks.jl

using LightGraphs

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

function closed_triad_removal(train_graph::SimpleGraph,
                              test_graph::SimpleGraph,
                              budget)
    # removable = begin
    #     s = Set()
    #     for e in edges(train_graph)
    #         for n in neighbors(test_graph, e[1])
    #         end
    #     end
    #     s
    # end

    train_graph = copy(train_graph)
    
    for b in 1:budget

        println(b)
        
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
    end

    train_graph
end

