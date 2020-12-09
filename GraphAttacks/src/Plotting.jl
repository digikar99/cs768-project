module Plotting

using GraphAttacks, LightGraphs
import ...CiteSeer
import ...SRW
import Random
cs = CiteSeer
ga = GraphAttacks


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
        if budgets[iter] == 0
            put!(channel,SimpleGraph(train))
            iter += 1
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

function random_add(train::SimpleGraph,test::SimpleGraph,budgets)
    flips=0
    iter=1
    nv_=nv(train)
    Channel() do channel
        if budgets[iter] == 0
            put!(channel,SimpleGraph(train))
            iter += 1
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
    Channel() do channel
        while true

            if iter > length(budgets) || flips>=maximum(budgets) break end
            if budgets[iter] == 0
                put!(channel,SimpleGraph(train))
                iter += 1
            end
            add_or_del=rand(1:2)
            if add_or_del==1
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
            else
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
                if !has_edge(train_graph, u, v) && u != v
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

# Dictionary from dataset to the permissible methods
permissible_methods = Dict(
    "Scale Free (small)" => ["Adamic Adar", "Katz", "Deep Walk"],
    "Scale Free (large)" => ["Adamic Adar", "Katz", "Deep Walk"],
    "CiteSeer" => ["Adamic Adar", "Katz", "Deep Walk"] # TODO: SRW
)

markers = Dict(
    "Adamic Adar" => "o",
    "Katz" => "v",
    "Deep Walk" => ">"
)

method_predictors = Dict(
    "Adamic Adar" => (perturbed_graph, per_node) -> predict(
        perturbed_graph,
        adamic_adar,
        per_node = per_node
    ),
    "Katz" => (perturbed_graph, per_node) -> predict(
        perturbed_graph,
        katz,
        per_node = per_node
    ),
    "Deep Walk" => (perturbed_graph, per_node) -> begin
        embeddings, _, _, _ =
            deepwalk_embedding(
                Array(adjacency_matrix(perturbed_graph)),
                5, # window_size
                min(32, nv(perturbed_graph)) # dim; nv of original and perturbed graph is same
                # min(32, nv(perturbed_graph)) # dim; nv of original and perturbed graph is same
            )
        predict_using_embeddings(perturbed_graph, embeddings, per_node)
    end,
    # "SRW" => (perturbed_graph, per_node) -> begin
    # println("================== $perturbed_graph ================================")
    # fcon = "/home/shubhamkar/ram-disk/citeseer/citeseer.content"
    # features = cs.read_features(fcon, tg, 100)
    # srw = SRW.make_SRW(tg, features, seed=9)
    # SRW.fit(srw, )
    # end

)

dataset_initializer = Dict(
    "Scale Free (small)" => () -> begin
    Random.seed!(0)
    scale_free(100, 1000, 2)
    end,
    "Scale Free (large)" => () -> scale_free(600, 10000, 2),
    "CiteSeer" => () -> begin
    mg = cs.read_into_graph("/home/shubhamkar/ram-disk/citeseer/citeseer.cites")[1]
    lg = cs.trim(mg, 5)
    tg = cs.extract_largest_connected_component(lg, true)
    end
)

katz_beta = 0.001

attacker_lambdas = Dict(
    "Random Deletion" => (train, test, budgets) ->
        random_del(SimpleGraph(train), budgets),
    "Random Addition" => (train, test, budgets) ->
        random_add(SimpleGraph(train), test, budgets),
    "Random Flips" => (train, test, budgets) ->
        random_flips(SimpleGraph(train), test, budgets),
    "CTR" => (train, test, budgets) ->
        closed_triad_removal(SimpleGraph(train), test, budgets),
    "OTC" => (train, test, budgets) ->
        open_triad_creation(SimpleGraph(train), test, budgets),
    "Greedy Katz" => (train, test, budgets) ->
        greedy_katz(SimpleGraph(train), test, budgets, katz_beta),
    "Node Embedding Attack" => (train, test, budgets) ->
        node_embedding_attack(SimpleGraph(train), test, budgets, min(32, nv(train)))
)


function emit_evaluation_data(dataset::String, attack_method;
                              budgets=[10 20 30 40 50],
                              train_fraction=0.8, seed=0)

    map_scores = Dict(
        "Adamic Adar" => [],
        "Katz" => [],
        "Deep Walk" => [],
        "SRW" => []
    )
    ap_scores = Dict(
        "Adamic Adar" => [],
        "Katz" => [],
        "Deep Walk" => [],
        "SRW" => []
    )
    katz_sim_scores = []

    full_graph  = dataset_initializer[dataset]()
    train, test = create_train_test_graph(full_graph, train_fraction, seed)
    print("Train: "); display(train)
    print("Test:  "); display(test)
    println("Attack Method: $attack_method")

    pred = nothing
    i    = 1
    for perturbed_graph in attacker_lambdas[attack_method](train, test, budgets)
        print("Perturbed Graph $i (budget $(budgets[i])): "); display(perturbed_graph)
        i += 1
        for method in permissible_methods[dataset]
            pred = method_predictors[method](perturbed_graph, true)
            push!(
                map_scores[method],
                ga.evaluate(perturbed_graph, test, pred, average_precision, per_node = true)
            )
            if method=="Katz" || method=="Random Deletion"
                push!(
                    katz_sim_scores,
                    katz_sum_scorer(
                        test,
                        float(Matrix(LightGraphs.LinAlg.adjacency_matrix(perturbed_graph))),
                        katz_beta
                    )
                )
            end
            pred = method_predictors[method](perturbed_graph, false)
            push!(
                ap_scores[method],
                ga.evaluate(perturbed_graph, test, pred, average_precision, per_node = false)
            )
        end
    end
    map_scores, ap_scores, katz_sim_scores
end

function print_for_pyplot(budgets, scores)

    print("Budget ")
    for budget in budgets
        print("$budget ")
    end
    print("\n")

    for method in keys(scores)
        println("$method")
        for score in scores[method]
            print("$score ")
        end
        print("\n")
    end
end

# p = Plotting
# p.emit_evaluation_data("CiteSeer", train_fraction = 0.85)

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
    g = begin
        if dataset=="scale_free"
            static_scale_free(100,1000,2)
        else
            create_simple_graph("//home/chitrank/cs768_datasets/datasets/GRQ_test_0.net")
        end
    end
    dim=min(32,nv(g))
    window_size=5

    budgets = [10*(1:5)...]
    train_fraction = 0.6
    train, test = create_train_test_graph(g,train_fraction)

    function Random_del()
        method_perturbed_graphs = random_del(SimpleGraph(train), budgets)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
        for perturbed_graph in method_perturbed_graphs
            println(perturbed_graph)
            # println("here!!!!!")
            pred    = predict(perturbed_graph, adamic_adar,   per_node=per_node)
            push!(acc_perturbed_AA,ga.evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
            push!(acc_perturbed_NE_sim,ga.evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
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
            push!(acc_perturbed_AA,ga.evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
            push!(acc_perturbed_NE_sim,ga.evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
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
            push!(acc_perturbed_AA,ga.evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
            push!(acc_perturbed_NE_sim,ga.evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
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
            println("CTR: $perturbed_graph")
            pred    = predict(perturbed_graph, adamic_adar, per_node=per_node)
            push!(acc_perturbed_AA,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
            push!(acc_perturbed_NE_sim,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
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
            println("OTC: $perturbed_graph")
            pred    = predict(perturbed_graph, adamic_adar, per_node=per_node)
            push!(acc_perturbed_AA,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
            push!(acc_perturbed_NE_sim,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
        end
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on OTC_perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on OTC perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on OTC perturbed",">")
    end
    function Katz()
        method_perturbed_graphs = greedy_katz(SimpleGraph(train), test, budgets)
        acc_perturbed_AA   = []
        acc_perturbed_katz = []
        acc_perturbed_NE_sim = []
        for perturbed_graph in method_perturbed_graphs
            pred    = predict(perturbed_graph, adamic_adar, per_node=per_node)
            push!(acc_perturbed_AA,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_= deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
            push!(acc_perturbed_NE_sim,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
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
            push!(acc_perturbed_AA,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            pred    = predict(perturbed_graph, katz,  per_node= per_node)
            push!(acc_perturbed_katz,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
            embeddings,_,_,_=deepwalk_embedding(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
            pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
            push!(acc_perturbed_NE_sim,ga.evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
        end
        # println("here")
        Add(allvals,plot_labels,acc_perturbed_AA,"AA on NEA-perturbed","o")
        Add(allvals,plot_labels,acc_perturbed_katz,"Katz on NEA-perturbed","v")
        Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on NEA-perturbed",">")
    end
    methods_to_methods_call=Dict(
        "CTR"=>CTR,
        "OTC"=>OTC,
        "Random_del"=>Random_del,
        "Random_add"=>Random_add,
        "Random_flips"=>Random_flips,
        "NEA"=>NEA
    )
    perturb_methods=["CTR","Random_add"]

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

end # module Plotting
