module GraphAttacks

using LightGraphs
using Random

function create_simple_graph(filename="/home/shubhamkar/ram-disk/datasets/FBK_full.net")::SimpleGraph
    # In accordance with the format at https://noesis.ikor.org/datasets/link-prediction
    open(filename) do f
        num_vertices = parse(Int, split(readline(f))[2])
        for i=1:num_vertices+1 readline(f); end
        readline(f)
        readline(f)
        g = SimpleGraph(num_vertices)
        while !eof(f)
            split_line = split(readline(f))
            u = parse(Int, split_line[1])
            v = parse(Int, split_line[2])
            add_edge!(g, u, v)
        end
        g
    end
end

function create_train_test_graph(graph::SimpleGraph,
                                 train_fraction::AbstractFloat = 0.8,
                                 seed::Int = 0)
    train = SimpleGraph(nv(graph))
    test  = SimpleGraph(nv(graph))
    Random.seed!(seed)
    for u = 1:nv(graph)
        u_neighbors = neighbors(graph, u)
        shuffled    = u_neighbors[randperm(length(u_neighbors))]
        num_training_edges = Int(ceil(length(u_neighbors)*train_fraction))
        # println(u, " ", num_training_edges, " ", shuffled)
        for i = num_training_edges+1 : length(u_neighbors)
            add_edge!(test,  u, shuffled[i])
        end
        for i = 1 : num_training_edges
            if !has_edge(test, u, shuffled[i])
                add_edge!(train, u, shuffled[i])
            end
        end
    end
    train, test
end

function adamic_adar(train_graph::SimpleGraph, u::Int, v::Int)::AbstractFloat
    score = 0
    for n in common_neighbors(train_graph, u, v)
        # computing a neighbor_length list before hand doesn't seem to be
        # making a big difference in efficiency
        n_len = length(neighbors(train_graph, n))
        score += 1 / log(n_len)
    end
    score
end

"""
Expects that sorted_v is a list of vertices corresponding to a pre-computed ranked_list
Intended to be used by evaluate
"""
function average_precision(test_graph::SimpleGraph,
                           ranked_list::Vector)
    num_edges_so_far = 0
    precision_sum    = 0.0
    num_pairs_so_far = 0
    for (u, v, _) in ranked_list
        num_pairs_so_far += 1
        if has_edge(test_graph, u, v)
            num_edges_so_far += 1
            precision_sum += num_edges_so_far/num_pairs_so_far
        end
    end
    if num_edges_so_far == 0
        nothing
    else
        precision_sum / num_edges_so_far
    end
end

function predict(train_graph::SimpleGraph, scorer, per_node::Bool=true)
    g = train_graph
    if per_node
        predictions = Dict()
        # should "export JULIA_NUM_THREADS=n" in .bashrc to take advantage
        # TODO: If needed, speed it up using type declarations?
        Threads.@threads for u in 1:nv(train_graph)
            predictions[u] = [
                (u, v, scorer(g,u,v)) for v in 1:nv(train_graph)
                if !has_edge(train_graph, u, v)
            ]
            sort!(predictions[u], by = x -> x[3], rev = true)
            if u%100 == 0 println("Processed $u nodes") end
        end
    else
        predictions_dict={}
        Threads.@threads for u in 1:nv(train_graph)
            predictions_dict[u]=[
                    (u, v, scorer(g,u,v)) for v in u+1:nv(train_graph)
                    if !has_edge(train_graph, u, v)
                ]
            end
        predictions=[predictions_dict[k] for k in keys(predictions_dict)]
        predictions=collect(Iterators.flatten(predictions))
        sort!(predictions, by = x -> x[3], rev = true)
    end        

    predictions
end

"
predictions -> a Dict mapping each node to the ranked_list with
               each entry of the form (u, v, score)
metric      -> a function that takes test_graph and a ranked_list as input and returns a score
"
function evaluate(train_graph::SimpleGraph,
                  test_graph::SimpleGraph,
                  predictions::Dict,
                  metric::Function,
                  per_node::Bool=true)
    total_result = 0.0
    num_nodes    = 0
    if per_node
        for u = 1:nv(train_graph)
            result = metric(test_graph, predictions[u])
            if result != nothing
                num_nodes    += 1
                total_result += result
            end
        end
        total_result/num_nodes
    else
        metric(test_graph,predictions)
    end
end

include("CTR.jl")

# Example Usage:
per_node=false
g           = create_simple_graph("//home/chitrank/datasets_cs768_project/GRQ_test_0.net")
train, test = create_train_test_graph(g)
pred        = predict(train, adamic_adar,per_node)
println(evaluate(train, test, pred, average_precision,per_node))
ctr         = closed_triad_removal(train, test, 10
ctr_pred    = predict(ctr, adamic_adar,per_node)
println(evaluate(ctr, test, ctr_pred, average_precision,per_node))
end


