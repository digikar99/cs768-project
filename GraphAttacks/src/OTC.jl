# This file is a part of GraphAttacks.jl

using LightGraphs
# include("utils.jl")


function open_triad_creation(train_graph::SimpleGraph,test_graph::SimpleGraph, budgets)
    iter=1
    function candidate_non_edges()
        list=[]
        for u in 1:nv(train_graph)
            for v in u:nv(train_graph)
                if !has_edge(train_graph,u,v) && !has_edge(test_graph,u,v)
                    block=0
                    for w in neighbors(test_graph,v)
                        if has_edge(train_graph,u,w)
                            block=1
                            break
                        end
                    end
                    if block==0
                        for w in neighbors(test_graph,u)
                            if has_edge(train_graph,v,w)
                                block=1
                                break
                            end
                        end
                    end
                    if block==0 push!(list,(u,v)) end
                end
            end
        end
        list
    end

    Channel() do channel
        candidate_non_edges_list=candidate_non_edges()

        if budgets[iter] == 0
            put!(channel,SimpleGraph(train_graph))
            iter += 1
        end

        for b in 1:maximum(budgets)

            scores = begin
                d = Dict()
                for e in candidate_non_edges_list
                    d[sort!([e[1], e[2]])] = 0 # assumes src <= dst
                end
                d
            end

            non_edge_with_max_score = begin
                for e in candidate_non_edges_list
                    u = e[1]
                    v = e[2]
                    block=0
                    for w in neighbors(test_graph,v)
                        if has_edge(train_graph,u,w)
                            block=1
                            break
                        end
                    end
                    # println("here")
                    if block==0
                        for w in neighbors(test_graph,u)
                            if has_edge(train_graph,v,w)
                                block=1
                                break
                            end
                        end
                    end
                    # println("here")
                    if block==1
                        deleteat!(candidate_non_edges_list, findall(y->y==(u,v),candidate_non_edges_list))
                        # delete!(scores, [u,v])
                    else
                        nu=neighbors(train_graph, u)
                        nv=neighbors(train_graph, v)
                        scores[sort!([u,v])]=length(setdiff(union(nu,nv), intersect(nu,nv)))
                    end
                end
                # println("here",u,v)
                if length(candidate_non_edges_list)>0
                    argmaximum(e->scores[[e[1],e[2]]], candidate_non_edges_list)
                else
                    nothing
                end
            end
            # println("chifnsfekjdsn")
            if non_edge_with_max_score==nothing
                error("Run out of good candidate non edges")
            end

            u, v = non_edge_with_max_score
            add_edge!(train_graph, u, v)
            deleteat!(candidate_non_edges_list, findall(y->y==(u,v),candidate_non_edges_list))
            # println("($u, $v) edge added: $train_graph")
            if iter<=length(budgets) && b==convert(Int,budgets[iter])
                put!(channel,SimpleGraph(train_graph))
                iter+=1
            end
        end
    end
end

