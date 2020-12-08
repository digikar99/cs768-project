using LinearAlgebra
using LightGraphs
# include("utils.jl")
# mutable struct NodeEmbeddingAttacker
#     graph
#     adj_matrix
#     degree_matrix
#     n_flips
#     dim
#     window_size
#     candidates #assuming to be a dense Array

# end
# function make_NEA(graph,n_flips, dim, window_size)
# 	adj_matrix=graph.adjacency_matrix
# 	degree_array=sum(Array(adj_matrix),dims=1)

# 	# degree_matrix
# 	candidates=nothing
# 	NodeEmbeddingAttacker(
# 		graph,
# 		adj_matrix,
# 		n_flips,
# 		dim,
# 		window_size
# 		)
# end

# function node_embedding_attack(train_graph::SimpleGraph,budgets)
# 	perturbation_top_flips(nea.adj_matrix, nea.candidates, nea.n_flips, nea.dim, nea.window_size)
# end

function node_embedding_attack(train_graph::SimpleGraph,test_graph::SimpleGraph,budgets, dim=32, window_size=5)
    """Selects the top (n_flips) number of flips using our perturbation attack.
    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param dim: int
        Dimensionality of the embeddings.
    :param window_size: int
        Co-occurence window size.
    :return: np.ndarray, shape [?, 2]
        The top edge flips from the candidate set
    """
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
    	delta_w[i]=1-2*adj_matrix[candidates_I,candidates_J]
    end

    # generalized eigenvalues/eigenvectors
    degree_array=[sum(adj_matrix,dims=2)...]
    deg_matrix=Diagonal(degree_array)
    vals, vecs = eigen(Array(adj_matrix), deg_matrix)

    loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals, vecs, n_nodes, dim, window_size)
    Channel() do channel
    	sorted_candidates_idx=[sortperm(loss_for_candidates,rev=true)...]
    	for b in budgets
 		   top_flips = candidates[sorted_candidates_idx[1:b],:]
 		   flips_matrix=sparse([top_flips[:,1]...], [top_flips[:,2]...], delta_w[sorted_candidates_idx[1:b]])+
						sparse([top_flips[:,2]...], [top_flips[:,1]...], delta_w[sorted_candidates_idx[1:b]])	
 		   perturbed_matrix=adj_matrix+flips_matrix
 		   embedding,_,_,_=deepwalk_svd(Array(perturbed_matrix),window_size,dim)
 		   put!(channel,(SimpleGraph(perturbed_matrix),embedding))
 		end
 	end

    # return top_flips
end


function estimate_loss_with_delta_eigenvals(candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size)
    """Computes the estimated loss using the change in the eigenvalues for every candidate edge flip.
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips,
    :param flip_indicator: np.ndarray, shape [?]
        Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :param n_nodes: int
        Number of nodes
    :param dim: int
        Embedding dimension
    :param window_size: int
        Size of the window
    :return: np.ndarray, shape [?]
        Estimated loss for each candidate flip
    """
    len_candidates=size(candidates,1)
    loss_est = zeros(len_candidates)
    for x in 1:len_candidates
        i, j = candidates[x,:]	
        vals_est = vals_org + flip_indicator[x] * (
                2 * vecs_org[:,i] .* vecs_org[:,j] - vals_org .* (vecs_org[:,i].^ 2 + vecs_org[:,j].^2))

        vals_sum_powers = sum_of_powers(vals_est, window_size)

        loss_ij =sum(sort(vals_sum_powers .^ 2)[1:n_nodes - dim]).^0.5
        loss_est[x] = loss_ij
    end

    return loss_est
end





function deepwalk_svd(adj_matrix, window_size::Int, embedding_dim::Int, num_neg_samples=1, 
	sparse=false)
    sum_powers_transition = sum_of_powers_of_transition_matrix(adj_matrix, window_size)

    deg = [sum(adj_matrix,dims=1)...]
    deg[deg .== 0] .= 1
    inv_deg_matrix = Diagonal(1 ./ deg)

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
	"""Computes an embedding by selection the top (embedding_dim) largest singular-values/vectors.
	:param x: sp.csr_matrix or np.ndarray
	    The matrix that we want to embed
	:param embedding_dim: int
	    Dimension of the embedding
	:param sparse: bool
	    Whether to perform sparse operations
	:return: np.ndarray, shape [?, embedding_dim], np.ndarray, shape [?, embedding_dim]
	    Embedding matrices.
	"""
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
