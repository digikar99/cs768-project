module CiteSeer
# Preprocessing code for CiteSeer data downloaded from
# https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz

using LightGraphs, MetaGraphs

"""
Returns two values: the graph and a dictionary of mapping from paper-id to graph-node
"""
function read_into_graph(filename, num_nodes=nothing)
    text_to_id = Dict()
    id_to_text = Dict()
    graph      = SimpleDiGraph()

    function get_text_to_id(text_id)
        if !(text_id in keys(text_to_id))
            add_vertex!(graph)
            text_to_id[text_id]   = nv(graph)
            id_to_text[nv(graph)] = text_id
        end
        text_to_id[text_id]
    end
    
    open(filename) do f
        while !eof(f)
            split_line = split(readline(f))
            text_u = split_line[1]
            u      = get_text_to_id(text_u)
            text_v = split_line[2]
            v      = get_text_to_id(text_v)
            if u != v add_edge!(graph, u, v) end
        end
    end
    while num_nodes != nothing && nv(graph)>num_nodes
        rem_vertex!(graph, nv(graph))
    end

    # println(has_edge(graph, 1, 1))
    # Node labeling
    graph = MetaDiGraph(graph)
    set_indexing_prop!(graph, :label)
    for i = 1:nv(graph)
        set_prop!(graph, i, :label, id_to_text[i])
    end
    graph, text_to_id
end

function read_features(filename, text_to_id, num_features=nothing)
    features = nothing
    num_total_nodes = length(text_to_id)
    
    open(filename) do f
        split_line = split(readline(f))
        if num_features == nothing num_features = length(split_line[2:end-1]) end
        features = zeros(Bool, num_total_nodes, num_features)
    end

    open(filename) do f
        while !eof(f)
            # Should be possible to speed up by avoiding allocations
            split_line = split(readline(f))
            id = text_to_id[split_line[1]]
            for feature_id = 1:num_features
                features[id, feature_id] = parse(Bool, split_line[1+feature_id])
            end
        end
    end
    
    features
end

function get_connected_components(graph, undirected=false)
    # TODO: Complete the undirected version
    components = Set()
    
    if undirected
        graph = MetaGraph(graph)
    end
    
    for e in edges(graph)
        # print(src(e), " ", dst(e), " ")
        u = get_prop(graph, src(e), :label)
        v = get_prop(graph, dst(e), :label)
        g_u, g_v = nothing, nothing
        for g in components
            if ! isempty(filter_vertices(g, :label, u))
                g_u = g
            end
            break
        end
        for g in components
            if ! isempty(filter_vertices(g, :label, v))
                g_v = g
            end
            break
        end
        
        if g_u == nothing && g_v == nothing
            tmp_g = MetaDiGraph(2)
            set_indexing_prop!(tmp_g, :label)
            # println(u, " ", v, " ", tmp_g)
            set_prop!(tmp_g, 1, :label, u)
            set_prop!(tmp_g, 2, :label, v)
            push!(components, tmp_g)
        elseif g_u != nothing && g_v != nothing
            tmp_g = join(g_u, g_v)
            delete!(components, g_u)
            delete!(components, g_v)
            push!(components, tmp_g)
        elseif g_u != nothing
            add_vertex!(g_u)
            set_prop!(g_u, nv(g_u), :label, v)
            i_u = g_u[u, :label]
            i_v = g_u[v, :label]
            add_edge!(g_u, i_u, i_v)
        elseif g_v != nothing
            add_vertex!(g_v)
            set_prop!(g_v, nv(g_v), :label, u)
            i_u = g_v[u, :label]
            i_v = g_v[v, :label]
            add_edge!(g_v, i_u, i_v)
        end
    end
    components
end

function get_largest_component(components::Set)
    max_size = 0
    max_component = nothing
    for c in components
        if length(c) > max_size
            max_size = length(c)
            max_component = c
        end
    end
    max_component
end

end # module Citeseer
