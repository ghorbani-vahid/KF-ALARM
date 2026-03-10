function nodes = assign_neighbors(nodes, adj_matrix)
    num_nodes = numel(nodes);
    for i = 1:num_nodes
        nodes{i}.neighbors = find(adj_matrix(i,:));
    end
end