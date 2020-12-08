using GraphAttacks
using Plots
using LightGraphs
# can you check this? sclae free formulation seemsto be depcreceated
# V=100
# E=200
# graph=scale_free(V,E)
function random_add(train::SimpleGraph,test::SimpleGraph,budgets)
	flips=0
	iter=1
	nv_=nv(train)
	Channel() do channel
		u=rand(1:nv_)
		v=rand(1:nv_)
		while v==u || has_edge(test,u,v) || has_edge(train,u,v)
			v=rand(1:nv_)
			u=rand(1:nv_)
		end
		flips+=1
		if flips==budgets[iter]
			add_edge!(train,u,v)
            put!(channel,SimpleGraph(train))
            iter+=1
        end
    end
end

function random_flips(train::SimpleGraph,test::SimpleGraph,budgets)
	flips=0
	iter=1
	nv_=nv(train)
	Channel() do channel
		add_or_del=rand(1:2)
		if add_or_del==1
			u=rand(1:nv_)
			v=rand(1:nv_)
			while v==u || has_edge(test,u,v) || has_edge(train,u,v)
				v=rand(1:nv_)
				u=rand(1:nv_)
			end
			flips+=1
			if flips==budgets[iter]
				add_edge!(train,u,v)
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
			if flips==budgets[iter]
				rem_edge!(train,u,v)
	            put!(channel,SimpleGraph(train))
	            iter+=1
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
	per_node=true

	function Add(values_,plot_labels,value,plot_label)
		if length(value)>0
			push!(values_,value)
			push!(plot_labels,plot_label)
		end
		return nothing
	end
	g= begin
			if dataset=="scale_free"
				static_scale_free(10,30,2)
			else
				create_simple_graph("//home/chitrank/cs768_datasets/datasets/GRQ_test_0.net")
			end
	end
	dim=min(32,nv(g)) 
	window_size=5

	budgets=[2,4,6]
	train_fraction=0.6
	train, test = create_train_test_graph(g,train_fraction)

	function Random_del()
		method_perturbed_graphs = random_del(SimpleGraph(train), budgets)
		acc_perturbed_AA   = []
		acc_perturbed_katz = []
		acc_perturbed_NE_sim = []
		for perturbed_graph in method_perturbed_graphs
			pred    = predict(perturbed_graph, adamic_adar,   per_node=per_node)
			push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
			pred    = predict(perturbed_graph, katz,  per_node= per_node)
			push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
			embeddings,_,_,_=deepwalk_svd(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
			pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
			push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
		end
		Add(allvals,plot_labels,acc_perturbed_AA,"AA on random-del-perturbed")
		Add(allvals,plot_labels,acc_perturbed_katz,"Katz on random-del-perturbed")
		Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on random-del-perturbed")
	end

	function Random_add()
		method_perturbed_graphs = random_add(SimpleGraph(train), test,budgets)
		acc_perturbed_AA   = []
		acc_perturbed_katz = []
		acc_perturbed_NE_sim = []
		for perturbed_graph in method_perturbed_graphs
			pred    = predict(perturbed_graph, adamic_adar,   per_node=per_node)
			push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
			pred    = predict(perturbed_graph, katz,  per_node= per_node)
			push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
			embeddings,_,_,_=deepwalk_svd(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
			pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
			push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
		end
		Add(allvals,plot_labels,acc_perturbed_AA,"AA on random-add-perturbed")
		Add(allvals,plot_labels,acc_perturbed_katz,"Katz on random-add-perturbed")
		Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on random-add-perturbed")
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
			embeddings,_,_,_=deepwalk_svd(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
			pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
			push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node=per_node))
		end
		Add(allvals,plot_labels,acc_perturbed_AA,"AA on random-flips-perturbed")
		Add(allvals,plot_labels,acc_perturbed_katz,"Katz on random-flips-perturbed")
		Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on random-flips-perturbed")
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
			embeddings,_,_,_=deepwalk_svd(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
			pred    = predict_using_embeddings(perturbed_graph, embeddings,  per_node)
			push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,  per_node= per_node))
		end
		Add(allvals,plot_labels,acc_perturbed_AA,"AA on CTR-perturbed")
		Add(allvals,plot_labels,acc_perturbed_katz,"Katz on CTR-perturbed")
		Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on CTR-perturbed")
	end
	function OTC()
		method_perturbed_graphs = open_triad_creation(SimpleGraph(train), test, budgets)
		acc_perturbed_AA   = []
		acc_perturbed_katz = []
		acc_perturbed_NE_sim = []
		for perturbed_graph in method_perturbed_graphs
			pred    = predict(perturbed_graph, adamic_adar,   per_node)
			push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
			pred    = predict(perturbed_graph, katz,   per_node)
			push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
			embeddings,_,_,_=deepwalk_svd(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
			pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
			push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
		end
		Add(allvals,plot_labels,acc_perturbed_AA,"AA on OTC_perturbed")
		Add(allvals,plot_labels,acc_perturbed_katz,"Katz on OTC perturbed")
		Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on OTC perturbed")
	end
	function Katz()
		method_perturbed_graphs = greedy_katz(SimpleGraph(train), test, budgets)
		acc_perturbed_AA   = []
		acc_perturbed_katz = []
		acc_perturbed_NE_sim = []
		for perturbed_graph in method_perturbed_graphs
			pred    = predict(perturbed_graph, adamic_adar,   per_node)
			push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
			pred    = predict(perturbed_graph, katz,   per_node)
			push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
			embeddings,_,_,_=deepwalk_svd(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
			pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
			push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
		end
		Add(allvals,plot_labels,acc_perturbed_AA,"AA on Katz-greedy_perturbed")
		Add(allvals,plot_labels,acc_perturbed_katz,"Katz on Katz-greey perturbed")
		Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim Katz-greedy perturbed")
	end
	function NEA()
		method_perturbed_graphs = node_embedding_attack(SimpleGraph(train), test, budgets,dim)
		acc_perturbed_AA   = []
		acc_perturbed_katz = []
		acc_perturbed_NE_sim = []
		for perturbed_graph in method_perturbed_graphs
			pred    = predict(perturbed_graph, adamic_adar,   per_node)
			push!(acc_perturbed_AA,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
			pred    = predict(perturbed_graph, katz,   per_node)
			push!(acc_perturbed_katz,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
			embeddings,_,_,_=deepwalk_svd(Array(adjacency_matrix(perturbed_graph)),window_size,dim)
			pred    = predict_using_embeddings(perturbed_graph, embeddings,   per_node)
			push!(acc_perturbed_NE_sim,evaluate(perturbed_graph, test, pred, average_precision,   per_node))
		end
		Add(allvals,plot_labels,acc_perturbed_AA,"AA on NEA-perturbed")
		Add(allvals,plot_labels,acc_perturbed_katz,"Katz on NEA-perturbed")
		Add(allvals,plot_labels,acc_perturbed_NE_sim,"DW_sim on NEA-perturbed")
	end
	methods_to_methods_call=Dict(
		"CTR"=>CTR,
		"OTC"=>OTC,
		"Random_del"=>Random_del,
		"Random_add"=>Random_add,
		"Random_flips"=>Random_flips,
		"NEA"=>NEA
		)
	perturb_methods=["CTR","Random_del"]
	for method in perturb_methods
		methods_to_methods_call[method]()
	end
	println(allvals)
	println(plot_labels)


	out_file="$(dataset)_$(join(perturb_methods,"_"))_all.txt"
	open(out_file,"w") do io
		for v in budgets
			write(io,string(v));
			write(io," ");
		end

		write(io,"\n");
		for v in plot_labels
			write(io,v);
			write(io," ");
		end

		for i in 1:length(plot_labels)
			write(io,"\n");
			for v in allvals[i]
				write(io,string(v));
				write(io," ");
			end
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

main()