module CiteSeer
# Preprocessing code for CiteSeer data downloaded from
# https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz

using LightGraphs, MetaGraphs, LinearAlgebra

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

function read_features(filename, graph, num_features=nothing)
    features = nothing
    num_total_features = nothing
    num_total_nodes = nv(graph)

    open(filename) do f
        split_line = split(readline(f))
        num_total_features = length(split_line[2:end-1])
        if num_features == nothing num_features = length(split_line[2:end-1]) end
        features = zeros(Bool, num_total_nodes, num_total_features)
    end

    i = 0
    open(filename) do f
        while !eof(f) # && i < 5
            i += 1
            # Should be possible to speed up by avoiding allocations
            split_line = split(readline(f))
            label = split_line[1]
            # println("Line $label")
            if ! isempty(filter_vertices(graph, :label, label))
                # println("Reading $label")
                id = graph[label, :label]
                for feature_id = 1:num_total_features
                    features[id, feature_id] = parse(Bool, split_line[1+feature_id])
                end
            end
        end
    end

    # num_features should affect SVD

    F = svd(features)
    F.U[:, 1:num_features]
end

function read_last_features(filename, graph)
    features = nothing
    num_total_features = 6
    num_total_nodes = nv(graph)
    feature_id_dict = Dict()

    # open(filename) do f
    #     split_line = split(readline(f))
    #     num_total_features = length(split_line[2:end-1])
    #     if num_features == nothing num_features = length(split_line[2:end-1]) end
    features = zeros(Bool, num_total_nodes, num_total_features)
    # end

    i = 0
    open(filename) do f
        while !eof(f) # && i < 5
            i += 1
            # Should be possible to speed up by avoiding allocations
            split_line = split(readline(f))
            label = split_line[1]
            # println("Line $label")
            if ! isempty(filter_vertices(graph, :label, label))
                # println("Reading $label")
                id      = graph[label, :label]
                feature = split_line[end]
                feature_id = nothing
                if feature in keys(feature_id_dict)
                    feature_id = feature_id_dict[feature]
                else
                    feature_id = length(feature_id_dict)+1
                    feature_id_dict[feature] = feature_id
                end
                features[id, feature_id] = 1
            end
        end
    end

    features
end

function ensure_vertex!(graph, label)
    if isempty(filter_vertices(graph, :label, label))
        add_vertex!(graph)
        set_prop!(graph, nv(graph), :label, label)
    end
end

function get_label(graph, vertex) get_prop(graph, vertex, :label) end

function extract_largest_connected_component(graph::MetaDiGraph, undirected=false, k=1)
    new_g = nothing

    if undirected
        new_g = MetaGraph()
        graph = MetaGraph(graph)
    else
        new_g = MetaDiGraph()
    end
    set_indexing_prop!(new_g, :label)

    ccs = sort!(
        weakly_connected_components(graph),
        by = cc -> length(cc),
        rev=true
    )
    required = ccs[k]

    new_vertices = Set(required)
    for u in new_vertices
        label_u = get_label(graph, u)
        # The neighbors are a part of the connected component containing u
        for v in neighbors(graph, u)
            label_v = get_label(graph, v)
            ensure_vertex!(new_g, label_u)
            ensure_vertex!(new_g, label_v)
            add_edge!(new_g, u, v)
        end
    end

    new_g
end

function trim(graph, degree_lower_bound=2)
    graph = copy(graph)
    current_vertex_idx = 1
    while current_vertex_idx <= nv(graph)
        if degree(graph, current_vertex_idx) < degree_lower_bound
            rem_vertex!(graph, current_vertex_idx)
        else
            current_vertex_idx += 1
        end
    end
    graph
end

end # module CiteSeer
