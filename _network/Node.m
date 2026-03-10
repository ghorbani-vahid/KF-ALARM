classdef Node
    %NODE Represents a single node in the distributed sensor network.

    properties
        id           % Node ID
        x_position   % Local x-coordinate
        y_position   % Local y-coordinate
        model        % Node model
        truth        % Ground truth data
        meas         % Measurements
        est_current  % Current estimate
        neighbors    % List of neighboring node IDs
    end

    methods
        function obj = Node(id, x, y, iter_num, attack, num_nodes, method)
            %Node Constructor to initialize the Node class.
            obj.id = id;
            obj.x_position = x;
            obj.y_position = y;

            % Generate model based on node position and network size
            obj.model = obj.gen_model(x, y, num_nodes);

            % Generate ground truth using the model, iterations, and attack type
            obj.truth = obj.gen_truth(obj.model, iter_num, attack);

            % Generate measurements from the model and ground truth
            obj.meas = obj.gen_meas(obj.model, obj.truth, attack);

            % Run node filter algorithm to get current estimate
            obj.est_current = obj.run_node(...
                obj.model, obj.meas, obj.truth, ...
                1, [], {}, [], iter_num, [], obj.id, ...
                "false", obj.x_position, obj.y_position, method);

            % Initialize neighbors list (empty by default)
            obj.neighbors = [];
        end

        function model = gen_model(~, x, y, num_nodes)
            %gen_model Wrapper to call the external gen_model function.
            model = gen_model(x, y, num_nodes);
        end

        function truth = gen_truth(~, model, iter_num, attack)
            %gen_truth Wrapper to call the external gen_truth function.
            truth = gen_truth(model, iter_num, attack);
        end

        function meas = gen_meas(~, model, truth, attack)
            %gen_meas Wrapper to call the external gen_meas function.
            meas = gen_meas(model, truth, attack);
        end

        function est = run_node(~, model, meas, truth, k, tt, tt_lmb_update_pre, neighbor_nodes, iter_num, estt, id, s, x_off, y_off, method)
            %run_node Wrapper to call the external run_node function.
            est = run_node(...
                model, meas, truth, k, tt, tt_lmb_update_pre, ...
                neighbor_nodes, iter_num, estt, id, s, x_off, y_off, method);
        end

        function handles = plot_results(~, model, truth, meas, est)
            %plot_results Wrapper to call the external plot_results function.
            handles = plot_results(model, truth, meas, est);
        end
    end
end
