using LightGraphs
using Random

function create_simple_graph(filename="/home/shubhamkar/ram-disk/datasets/FBK_full.net")
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

function create_train_graph(graph, train_fraction = 0.8)
    train = SimpleGraph(nv(graph))
    for u in vertices(graph)
        u_neighbors = neighbors(graph, u)
        shuffled    = u_neighbors[randperm(length(u_neighbors))]
        num_training_edges = Int(ceil(length(u_neighbors)*train_fraction))
        for i = 1 : num_training_edges
            add_edge!(train, u, shuffled[i])
        end
    end
    train
end

"""
Returns a grouped_ranked_list on the basis of adamic_adar 
\"trained\" on the train_graph
"""
function adamic_adar(train_graph)
    g = train_graph
    predictions = Dict()
    n_len = [length(neighbors(g, n)) for n in vertices(train_graph)]
    function _adamic_adar(u,v)
        score = 0
        for n in common_neighbors(g, u, v)
            score += 1 / log(n_len[n])
        end
        score
    end
    # should "export JULIA_NUM_THREADS=n" in .bashrc to take advantage
    # TODO: If needed, speed it up using type declarations?
    Threads.@threads for u in 1:nv(train_graph)
        predictions[u] = [
            (u, v, _adamic_adar(u,v)) for v in 1:nv(train_graph)
            if !has_edge(train_graph, u, v)
        ]
        sort!(predictions[u], by = x -> x[3], rev = true)
        if u%100 == 0 println("Processed $u nodes") end
    end
    predictions
end
