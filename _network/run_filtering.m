function execution_times = run_filtering(nodes, iter_num, save_plots, method)
    num_nodes = numel(nodes);
    tt_lmb_update_pre = cell(1,num_nodes);
    execution_times   = zeros(1, iter_num);

    for iter = 1:iter_num
        fprintf('Iteration %d\n', iter);
        % Communication
        new_estimates = cell(1,num_nodes);
        neighbors     = cell(1,num_nodes);
        parfor i = 1:num_nodes
            nbrs = nodes{i}.neighbors;
            ests = [];
            for nid = nbrs
                ests = [ests, nodes{nid}.est_current];
            end
            new_estimates{i} = ests;
            neighbors{i}     = nodes(nbrs);
        end

        % Processing
        total_time_iter = 0;
        for i = 1:num_nodes
            tic;
            nn_struct = convertToStructArray(neighbors{i});
            nodes{i}.est_current = nodes{i}.run_node(...
                nodes{i}.model, nodes{i}.meas, nodes{i}.truth, iter, ...
                new_estimates{i}, tt_lmb_update_pre{i}, nn_struct, iter_num, ...
                nodes{i}.est_current, i, save_plots, nodes{i}.x_position, nodes{i}.y_position, method);
            elapsed = toc;
            total_time_iter = total_time_iter + elapsed;
            tt_lmb_update_pre{i} = nodes{i}.est_current.tt_lmb_update;
        end
        execution_times(iter) = total_time_iter / num_nodes;
        fprintf('Mean execution time: %.2f seconds\n', mean(execution_times(1:iter)));
    end
end