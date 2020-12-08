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
include("CiteSeer.jl")
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
function random_del(train::SimpleGraph, budgets)
    # new_g = copy(train)
    flips=0
    iter=1
    nv_=nv(train)
    Channel() do channel
        if minimum(budgets)==0
            put!(channel,train)
            iter+=1
        end

        while true
            if flips>=maximum(budgets)
                break
            end
            u=rand(1:nv_)
            v=rand(1:nv_)
            while v==u || !has_edge(train,u,v)
                v=rand(1:nv_)
                u=rand(1:nv_)
            end
            flips+=1
            rem_edge!(train,u,v)
            if flips==budgets[iter]
                put!(channel,SimpleGraph(train))
                iter+=1
            end
        end
    end
end


include("utils.jl")
include("CTR.jl")
include("OTC.jl")
include("node_embedding_attack.jl")
# using GraphAttacks
using Plots
# using LightGraphs
# can you check this? sclae free formulation seemsto be depcreceated
# V=100
# E=200
# graph=scale_free(V,E)
function random_add(train::SimpleGraph,test::SimpleGraph,budgets)
    flips=0
    # iter=1
    nv_=nv(train)
    iter=1

    Channel() do channel
        if minimum(budgets)==0
            put!(channel,train)
            iter+=1
        end
        while true
            if flips>=maximum(budgets)
                break
            end
            u=rand(1:nv_)
            v=rand(1:nv_)
            while v==u || has_edge(test,u,v) || has_edge(train,u,v)
                v=rand(1:nv_)
                u=rand(1:nv_)
            end
            flips+=1
            add_edge!(train,u,v)
            if flips==budgets[iter]
                put!(channel,SimpleGraph(train))
                iter+=1
            end
        end
    end
end

function random_flips(train::SimpleGraph,test::SimpleGraph,budgets)
    flips=0
    iter=1
    nv_=nv(train)
    edges_added=[]
    edges_deleted=[]
    Channel() do channel
        while true
            if flips>=maximum(budgets)
                break
            end
            add_or_del=rand(1:2)
            if add_or_del==1 #add_edge
                u=rand(1:nv_)
                v=rand(1:nv_)
                while v==u || has_edge(test,u,v) || has_edge(train,u,v) || (sort!([u,v]) in edges_deleted)
                    v=rand(1:nv_)
                    u=rand(1:nv_)
                end
                flips+=1
                add_edge!(train,u,v)
                push!(edges_added,sort!([u,v]))
                if flips==budgets[iter]
                    put!(channel,SimpleGraph(train))
                    iter+=1
                end
            else #delete edge
                u=rand(1:nv_)
                v=rand(1:nv_)
                while v==u || !has_edge(train,u,v) || (sort!([u,v]) in edges_added)
                    v=rand(1:nv_)
                    u=rand(1:nv_)
                end
                flips+=1
                rem_edge!(train,u,v)
                push!(edges_deleted,sort!([u,v]))
                if flips==budgets[iter]
                    put!(channel,SimpleGraph(train))
                    iter+=1
                end
            end
        end
    end
end



function cosine_sim(embeddings,dim_axis=2)
    norm_embeddings=embeddings./sqrt.(sum((embeddings.*embeddings),dims=dim_axis))
    norm_embeddings*transpose(norm_embeddings)
end
function predict_using_embeddings(train_graph::SimpleGraph,embeddings,
                 per_node::Bool=false)

    g = train_graph

    predictions  = nothing
    score_matrix = cosine_sim(embeddings)
    
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

# g           = create_simple_graph("/home/shubhamkar/ram-disk/datasets/GRQ_test_0.net")
function main()
    # dataset="GRQ"
    dataset="scale_free"
    # method="OTC"
    allvals=[]
    plot_labels=[]
    markers=[]
    per_node=true

    function Add(values_,plot_labels,value,plot_label,marker)
        if length(value)>0
            push!(values_,value)
            push!(plot_labels,plot_label)
            push!(markers,marker)
        end
        return nothing
    end
    g= begin
            if dataset=="scale_free"
                static_scale_free(100,1000,2)
            else
                create_simple_graph("//home/chitrank/cs768_datasets/datasets/GRQ_test_0.net")
            end
    end
    dim=min(32,nv(g)) 
    window_size=5

    budgets=[10*(0:5)...]
    train_fraction=0.8
    train, test = create_train_test_graph(g,train_fraction)

    function Random_del()
        method_perturbed_graphs = random_del(SimpleGraph(train), budgets)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
        for perturbed_graph in method_perturbed_graphs
            println(perturbed_graph)
            println("here!!!!!")
            pred    = predict(perturbed_graph, adamic_adar,   per_node=per_node)
            push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
            push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
        end
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on random-del-perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on random-del-perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on random-del-perturbed",">")
    end

    function Random_add()
            # println(" random add here!!!!!")
        method_perturbed_graphs = random_add(SimpleGraph(train), test,budgets)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
            # println(method_perturbed_graphs)

        for perturbed_graph in method_perturbed_graphs
            # println("here!!!!!")
            # println()
            println(perturbed_graph)
            # println()
            pred    = predict(perturbed_graph, adamic_adar,   per_node=per_node)
            push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
            push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
        end
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on random-add-perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on random-add-perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on random-add-perturbed",">")
    end

    function Random_flips()
        method_perturbed_graphs = random_flips(SimpleGraph(train), test,budgets)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
        for perturbed_graph in method_perturbed_graphs
            pred    = predict(perturbed_graph, adamic_adar,   per_node=per_node)
            push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
            push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
        end
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on random-flips-perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on random-flips-perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on random-flips-perturbed",">")
    end

    function CTR()
        method_perturbed_graphs = closed_triad_removal(SimpleGraph(train), test, budgets)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
        for perturbed_graph in method_perturbed_graphs
            pred    = predict(perturbed_graph, adamic_adar, per_node=per_node)
            push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
            push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
        end
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on CTR-perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on CTR-perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on CTR-perturbed",">")
    end
    function OTC()
        method_perturbed_graphs = open_triad_creation(SimpleGraph(train), test, budgets)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
        for perturbed_graph in method_perturbed_graphs
            pred    = predict(perturbed_graph, adamic_adar, per_node=per_node)
            push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
            push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
        end
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on OTC_perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on OTC perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on OTC perturbed",">")
    end
    function Katz()
        method_perturbed_graphs = greedy_katz(SimpleGraph(train), test, budgets,0.001)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
        for perturbed_graph in method_perturbed_graphs
            pred    = predict(perturbed_graph, adamic_adar, per_node=per_node)
            push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
            push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
        end
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on Katz-greedy_perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on Katz-greey perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim Katz-greedy perturbed",">")
    end
    function NEA()
        method_perturbed_graphs = node_embedding_attack(SimpleGraph(train), test, budgets,dim)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
        # println("here")
        println(method_perturbed_graphs)

        for perturbed_graph in method_perturbed_graphs
            # println("here")
            pred    = predict(perturbed_graph, adamic_adar, per_node=per_node)
            push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_=deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
            push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
        end
        # println("here")
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on NEA-perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on NEA-perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on NEA-perturbed",">")
    end
    methods_to_methods_call=Dict(
        "CTR"=>CTR,
        "OTC"=>OTC,
        "Katz"=>Katz,
        "Random_del"=>Random_del,
        "Random_add"=>Random_add,
        "Random_flips"=>Random_flips,
        "NEA"=>NEA
        )
    perturb_methods=["CTR","Random_del"]

    for method in perturb_methods
        println(methods_to_methods_call[method])
        methods_to_methods_call[method]()
    end
    println(allvals)
    println(plot_labels)


    out_file="$(dataset)_$(join(perturb_methods,"_"))_all.txt"
    open(out_file,"w") do io
        write(io,join(map(x->string(x),budgets),","));

        write(io,"\n");
        write(io,join(map(x->string(x),plot_labels),","));
        write(io,"\n");
        write(io,join(map(x->string(x),markers)," "));

        for i in 1:length(plot_labels)
            write(io,"\n");
            write(io,join(map(x->string(x),allvals[i]),","));
        end

    end

    # vals=zeros(length(budgets),2)
    # vals[:,1]=acc_perturbed_AA
    # vals[:,2]=acc_perturbed_katz
    # p=plot(budgets,vals,label=["AA","katz"],title="Acc vs budgets")
    # xlabel!(p,"budgets")
    # ylabel!(p,"accuracy (MAP)")
    # display(p)
end
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
