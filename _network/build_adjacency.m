function adj_matrix = build_adjacency(sensor_locations)
    distance_threshold = 1900;      % Meters
    num_nodes = size(sensor_locations,1);
    adj_matrix = zeros(num_nodes);
    for i = 1:num_nodes
        for j = i+1:num_nodes
            if norm(sensor_locations(i,:) - sensor_locations(j,:)) <= distance_threshold
                adj_matrix(i,j) = 1;
                adj_matrix(j,i) = 1;
            end
        end
    end
end