function nodes = initialize_nodes(sensor_locations, iter_num, attacked_sensor, attack_scenario, ghost_num, method)
    num_nodes = size(sensor_locations,1);
    x_coords  = sensor_locations(:,1);
    y_coords  = sensor_locations(:,2);
    nodes = cell(1, num_nodes);
    for i = 1:num_nodes
        attack.scenario  = 'none';
        attack.intensity = 0;
        attack.ghost_num = 0;
        if ismember(i, attacked_sensor)
            attack.scenario  = attack_scenario;
            attack.ghost_num = ghost_num;
        end
        nodes{i} = Node(i, x_coords(i), y_coords(i), iter_num, attack, num_nodes, method);
    end
end