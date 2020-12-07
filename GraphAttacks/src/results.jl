using GraphAttacks
using Plots
using LightGraphs
# can you check this? sclae free formulation seemsto be depcreceated
# V=100
# E=200
# graph=scale_free(V,E)
function cosine_sim(embeddings,dim_axis=1)
  norm_embeddings=embeddings./sqrt.(sum((embeddings.*embeddings),dims=dim_axis))
  transpose(norm_embeddings)*norm_embeddings
end

function predict_using_embeddings(train_graph::SimpleGraph,embeddings,
                                  per_node::Bool=PER_NODE)

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

g           = create_simple_graph("/home/shubhamkar/ram-disk/datasets/GRQ_test_0.net")
# g             = create_simple_graph("//home/chitrank/cs768_datasets/datasets/GRQ_test_0.net")
budgets       = [2 4 6 8]
train, test   = create_train_test_graph(g)
train1        = SimpleGraph(train)
pred_original = predict(train, adamic_adar, per_node = true)
acc_original  = evaluate(train, test, pred_original, average_precision, per_node = true)
ctr           = closed_triad_removal(train, test, budgets)
println(ctr)

acc_perturbed_AA   = []
acc_perturbed_katz = []
for perturbed_graph in ctr
    global acc_perturbed_AA, acc_perturbed_katz
    pred    = predict(perturbed_graph, adamic_adar, per_node = true)
    push!(
        acc_perturbed_AA,
        evaluate(perturbed_graph, test, pred, average_precision, per_node = true)
    )
    pred    = predict(perturbed_graph, katz, per_node = true)
    push!(
        acc_perturbed_katz,
        evaluate(perturbed_graph, test, pred, average_precision, per_node = true)
    )
end

println(acc_perturbed_AA)
println(acc_perturbed_katz)
labels=["AA","katz"]

open("results.txt","w") do io
    for v in budgets
        write(io,string(v));
        write(io," ");
    end

    write(io,"\n");
    for v in labels
        write(io,v);
        write(io," ");
    end

    write(io,"\n");
    for v in acc_perturbed_AA
        write(io,string(v));
        write(io," ");
    end

    write(io,"\n");
    for v in acc_perturbed_katz
        write(io,string(v));
        write(io," ");
    end
end


