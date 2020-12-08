# This file is a part of GraphAttacks.jl

using LightGraphs
# include("utils.jl")


function open_triad_creation(train_graph::SimpleGraph,test_graph::SimpleGraph, budget)
    # train_graph = copy(train_graph)
    iter=1
    function candidate_non_edges()
        list=[]
        # Channel() do channel
            for u in 1:nv(train_graph)
                for v in u:nv(train_graph)
                    if !has_edge(train_graph,u,v) && !has_edge(test_graph,u,v) && (degree(test_graph,u)>0 ||
                        degree(test_graph,v)>0)
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
                        if block==0
                            push!(list,(u,v))
                        end
                    end
                end
            end
        # end
        list
    end

    Channel() do channel
        candidate_non_edges_list=candidate_non_edges()

        for b in 1:maximum(budgets)

            println(b)
            
            scores = begin
                d = Dict()
                for e in candidate_non_edges_list
                    d[sort!([e[1], e[2]])] = 0 # assumes src <= dst
                end
                d
            end

            non_edge_with_max_score = 
                begin
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
                        if block==0
                            for w in neighbors(test_graph,u)
                                if has_edge(train_graph,v,w)
                                    block=1
                                    break
                                end
                            end
                        end
                        if block==1
                            deleteat!(candidate_non_edges_list, findall(y->y==(u,v),candidate_non_edges_list))
                        else
                            nu=neighbors(train_graph, u)
                            nv=neighbors(train_graph, v)
                            scores[sort!([u,v])]=length(setdiff(union(nu,nv), intersect(nu,nv)))                    
                        end
                    end
                    if length(candidate_non_edges_list)>0
                        argmaximum(e->scores[e], candidate_non_edges_list)
                    else
                        nothing
                    end
                end
            if edge_with_max_score==nothing
                error("Run out of good candidate non edges")
            end

            u, v = edge_with_max_score
            add_edge!(train_graph, u, v)
            if iter<=length(budgets) && b==convert(Int,budgets[iter])
                put!(channel,SimpleGraph(train_graph))
                iter+=1
            end
        end
    end
end

