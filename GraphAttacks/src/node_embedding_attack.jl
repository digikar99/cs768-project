using LinearAlgebra
using LightGraphs
using SparseArrays

function node_embedding_attack(train_graph::SimpleGraph,test_graph::SimpleGraph,budgets, dim=32, window_size=5)
    adj_matrix=adjacency_matrix(train_graph)
    candidates=
    		begin
    			l=[]
    			for u in 1:nv(train_graph)
    				for v in u:nv(train_graph)
    					if !has_edge(test_graph,u,v)
    						push!(l, [u,v])
    					end
    				end
    			end
    			l
    		end
    candidates_I=map(x->x[1],candidates)
    candidates_J=map(x->x[2],candidates)


    n_nodes = adj_matrix.m # m gives the no of rows, here it is the number of nodes
    
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    # delta_w = 1 - 2 * [(adj_matrix[candidates[:, 0], candidates[:, 1]].A1
    len_candidates=size(candidates,1)
    delta_w=zeros(len_candidates)
    for i in 1:len_candidates
    	delta_w[i]=1 .- 2 .* adj_matrix[candidates_I[i],candidates_J[i]]
    end

    # generalized eigenvalues/eigenvectors
    degree_array=[sum(adj_matrix,dims=2)...]
    deg_matrix=Array(Diagonal(degree_array))
    vals, vecs = eigen(Array(adj_matrix), deg_matrix)

    loss_for_candidates = loss_estimate(candidates, delta_w, vals, vecs, n_nodes, dim, window_size)
        	# println("here nea")
    Channel() do channel
        	# println("here nea")
    	sorted_candidates_idx=[sortperm(loss_for_candidates,rev=true)...]
    	for b in budgets
        	# println("here nea")
 		   top_flips = candidates[sorted_candidates_idx[1:b],:]
 		   top_flips_I=[map(x->x[1],top_flips)...]
 		   top_flips_J=[map(x->x[2],top_flips)...]
        	# println("here nea")
 		   flips_matrix=sparse(top_flips_I, top_flips_J,delta_w[sorted_candidates_idx[1:b]],n_nodes,n_nodes)+
						sparse(top_flips_J, top_flips_I, delta_w[sorted_candidates_idx[1:b]],n_nodes,n_nodes)	
        	# println("here nea")
 		   perturbed_matrix=adj_matrix+flips_matrix
 		   # embedding,_,_,_=deepwalk_embedding(Array(perturbed_matrix),window_size,dim)
        	# println("here nea")

 		   put!(channel,SimpleGraph(perturbed_matrix))
 		end
 	end

    # return top_flips
end


function loss_estimate(candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size)
    len_candidates=size(candidates,1)
    loss_est = zeros(len_candidates)
    for x in 1:len_candidates
        i, j = candidates[x]	
        vals_est = vals_org + flip_indicator[x] * (
                2 * vecs_org[:,i] .* vecs_org[:,j] - vals_org .* (vecs_org[:,i].^ 2 + vecs_org[:,j].^2))

        vals_sum_powers = sum_power_x(vals_est, window_size)

        loss_ij =sum(sort(vals_sum_powers .^ 2 , dims=2)[1:n_nodes - dim]).^0.5
        loss_est[x] = loss_ij
    end

    return loss_est
end





function deepwalk_embedding(adj_matrix, window_size::Int, embedding_dim::Int, num_neg_samples=1, 
	sparse=false)
    sum_powers_transition = sum_power_transition(adj_matrix, window_size)

    deg = [sum(adj_matrix,dims=1)...]
    deg[deg .== 0] .= 1
    inv_deg_matrix = Array(Diagonal(1 ./ deg))

    volume = sum(adj_matrix)

    M = sum_powers_transition*inv_deg_matrix * volume / (num_neg_samples * window_size)

    log_M = M
    log_M[M .> 1] = log.(log_M[M .> 1])
    log_M = log_M.*(M .> 1.)

    if !sparse
        log_M = Array(log_M)
    end

    Fu= svd_embedding(log_M, embedding_dim, sparse)
    Fv=nothing
    loss=nothing
    # loss = norm(Fu*transpose(Fv) - log_M,p=2)

    return Fu, Fv,loss, log_M	
end 

function svd_embedding(x, embedding_dim, sparse=false)
	if sparse
	    F = svd(Array(x))
	else
	    F = svd(x)
	end
	U,S,V=F.U,F.S,F.V

	S = Diagonal(S[1:embedding_dim])
	Fu = U[:,1:embedding_dim]*(sqrt.(S))
	# Fv = V[:,1:embedding_dim]*(sqrt.(S))
	# Fv = np.sqrt(S).dot(V)[:embedding_dim, :].T

	return Fu
end


function sum_power_x(x, power)
    
    n = size(x,1)
    sum_powers = zeros(power, n)
    for i in 1:power
        sum_powers[i,:] .= x .^ i
    end

    return sum(sum_powers,dims=1)
end


function sum_power_transition(adj_matrix, pow)

    deg = [sum(adj_matrix,dims=1)...]
    deg[deg .== 0] .= 1
    inv_deg_matrix = Array(Diagonal(1 ./ deg))
    transition_matrix = inv_deg_matrix * adj_matrix

    sum_of_powers = transition_matrix
    last_ = transition_matrix
    for i in 1:pow-1
        last_ = last_*transition_matrix
        sum_of_powers += last_
    end

    return sum_of_powers
end

