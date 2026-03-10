function plot_network_topology(num_nodes, adj_matrix, x_coords, y_coords)
    figure; hold on;
    scatter(x_coords, y_coords, 100, 'filled');
    text(x_coords+1, y_coords, arrayfun(@num2str,1:num_nodes,'UniformOutput',false));
    for i = 1:num_nodes
        for j = i+1:num_nodes
            if adj_matrix(i,j)
                plot([x_coords(i) x_coords(j)], [y_coords(i) y_coords(j)], 'k-','LineWidth',1.5);
            end
        end
    end
    title('Network Topology'); xlabel('X Coordinate'); ylabel('Y Coordinate');
    grid on; axis equal; hold off;
    save_dir = fullfile('results','network'); if ~exist(save_dir,'dir'), mkdir(save_dir); end
    saveas(gcf, fullfile(save_dir,'network_topology.png'));
    disp('Network topology figure saved.');
end