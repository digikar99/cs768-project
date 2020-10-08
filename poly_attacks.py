def ctr(graph,budget,removable,hidden):
	"""
	input- graph (networkx), budget (the maximum no of edges in removable that can be remooved),
			removable (list of edges which are remove-able) and hidden (the list of all hidden/undeclared relations
			which are non-edges in graph)
	returns- the set of optimal removal edges according to the budget. Prints error and exit if the size of optimal 
	edges is less than the budget
	"""

	score_removable={i:0 for i in removable}

	hiddens_per_node={i:0 for i in G}
	for (u_h,v_h) in hidden:
		hiddens_per_node[u_h]+=1
		hiddens_per_node[v_h]+=1
		CN_uh_vh=nx.common_neighbors(G,u_h,v_h)
		for w in CN_uh_vh:
			a=min(u_h,w)
			b=max(u_h,w)
			if (a,b) in score_removable:
				score_removable[(a,b)]+=1
			a=min(v_h,w)
			b=max(v_h,w)
			if (a,b) in score_removable:
				score_removable[(a,b)]+=1
	for (u,v) in removable:
		if score_removable[(u,v)]==0:
			score_removable.pop((u,v),None)
	removed=[]

	for i in range(budget):
		max_score_nodes=None
		max_score=0
		for (u,v) in score_removable:
			if max_score<score_removable[(u,v)]:
				max_score=score_removable[(u,v)]
				max_score_nodes=(u,v)
		if max_score_nodes is None:
			print("ERR!!! All scores are 0")
			exit(1)
		removed.append(max_score_nodes)
		u,v=max_score_nodes
		for (u_h,v_h) in hidden:
			if u==u_h:
				minn=min(v,v_h)
				maxn=max(v,v_h)
			elif v==v_h:
				minn=min(u,u_h)
				maxn=max(u_h,u)
			if (minn,maxn) in score_removable:
				score_removable[(minn,maxn)]-=1
			if score_removable[(minn,maxn)]==0:
				score_removable.pop((minn,maxn),None)
	return removed


				







