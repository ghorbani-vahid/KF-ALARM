%% File: main.m
clear;
rng(42);                        % Reproducibility
iter_num = 100;                 % Number of filtering iterations

% Select filtering method and get sensor layout
method = input('Filtering method (ALARM / ST): ', 's');
sensor_locations = define_sensor_locations(method);
adj_matrix = build_adjacency(sensor_locations);

% Configure attack settings
[attacked_sensor, attack_scenario, ghost_num] = configure_attack(method);
    disp('Attacked sensors:'), disp(attacked_sensor);

save_flag     = input('Save plots? (true/false): ', 's');

% Initialize node structures and neighbors
nodes = initialize_nodes(sensor_locations, iter_num, attacked_sensor, attack_scenario, ghost_num, method);
nodes = assign_neighbors(nodes, adj_matrix);

    
% Run the filtering iterations
execution_times = run_filtering(nodes, iter_num, save_flag, method);

% Optionally save timing results
if strcmp(input('Save execution times to MAT file? (y/n): ', 's'), 'y')
    save('execution_times.mat', 'execution_times');
    fprintf('Execution times saved. Total: %.2f seconds\n', sum(execution_times));
end