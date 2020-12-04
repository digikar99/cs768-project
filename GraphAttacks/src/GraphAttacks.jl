module GraphAttacks

using LightGraphs
using LinearAlgebra
using Random
import ExportAll

scale_free = LightGraphs.SimpleGraphs.static_scale_free

# TODO: figure out how to change this from other modules, eg. Main/Base
PER_NODE    = false # or true

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

include("Heuristics.jl")

include("SRW.jl")

function predict(train_graph::SimpleGraph, scorer::Function;
                 per_node::Bool=PER_NODE, beta::AbstractFloat=0.001)

    g = train_graph

    predictions  = nothing
    score_matrix = if scorer == katz
        scorer(train_graph, beta)
    else
        scorer(train_graph)
    end
    
    if per_node
        predictions = Dict()
        # should "export JULIA_NUM_THREADS=n" in .bashrc to take advantage
        # TODO: If needed, speed it up using type declarations?
        for u in 1:nv(train_graph)
            predictions[u] = [
                (u, v, score_matrix[u,v]) for v in 1:nv(train_graph)
                if !has_edge(train_graph, u, v)
            ]
            sort!(predictions[u], by = x -> x[3], rev = true)
            # if u%100 == 0 println("Processed $u nodes") end
        end
    else
        predictions=[]
        for u in 1:nv(train_graph)
            append!(predictions,[
                (u, v, score_matrix[u,v]) for v in u+1:nv(train_graph)
                if !has_edge(train_graph, u, v)
            ])
        end
        sort!(predictions, by = x -> x[3], rev = true)
    end
    predictions
end

"""
predictions -> a Dict mapping each node to the ranked_list with
               each entry of the form (u, v, score)
metric      -> a function that takes test_graph and a ranked_list as input
               and returns a score
"""
function evaluate(train_graph::SimpleGraph,
                  test_graph::SimpleGraph,
                  predictions,
                  metric::Function;
                  per_node::Bool=PER_NODE)
    if per_node
        total_result = 0.0
        num_nodes    = 0
        for u = 1:nv(train_graph)
            result = metric(test_graph, predictions[u])
            if result != nothing
                num_nodes    += 1
                total_result += result
            end
        end
        total_result/num_nodes
    else
        metric(test_graph, predictions)
    end
end

include("Greedy.jl")

"""
Returns a new graph, formed by randomly deleting edges from the graph
train : graph, intended to be the training graph
n_del : number of edges to delete
"""
function random_del(train::SimpleGraph, n_del::Int)
    new_g = copy(train)

    del_vec = Set(
        Random.shuffle(
            collect(1:ne(train))
        )[1:n_del]
    )

    idx = 1
    for e in edges(new_g)
        if idx in del_vec
            rem_edge!(new_g, e)
        end
        idx += 1
    end
    new_g
end


include("CTR.jl")

ExportAll.@exportAll()

# Example Usage:
# g           = create_simple_graph("/home/shubhamkar/ram-disk/datasets/GRQ_test_0.net")
# g           = create_simple_graph("//home/chitrank/datasets_cs768_project/GRQ_test_0.net")
# train, test = create_train_test_graph(g)
# pred        = predict(train, adamic_adar, per_node = true)
# evaluate(train, test, pred, average_precision, per_node = true)
# ctr         = closed_triad_removal(train, test, 10)
# ctr_pred    = predict(ctr, adamic_adar, per_node = true)
# evaluate(ctr, test, ctr_pred, average_precision, per_node = true)
end
