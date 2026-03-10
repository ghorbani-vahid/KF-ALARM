
function handles= plot_results(model,truth,meas,est,unique_id, method)

set(0, 'DefaultTextFontName', 'Times New Roman', 'DefaultAxesFontName', 'Times New Roman', 'DefaultLegendFontName', 'Times New Roman');

  results_folder = fullfile('results',  method, num2str(unique_id)); % Create folder path
    if ~exist(results_folder, 'dir')
        mkdir(results_folder); % Create folder if it doesn't exist
    end
    
[X_track,k_birth,k_death]= extract_tracks(truth.X,truth.track_list,truth.total_tracks);

labelcount= countestlabels();
colorarray= makecolorarray(labelcount);
est.total_tracks= labelcount;
est.track_list= cell(truth.K,1);
for k=1:truth.K
    for eidx=1:size(est.X{k},2)
        est.track_list{k} = [est.track_list{k} assigncolor(est.L{k}(:,eidx))];
    end
end
[Y_track,l_birth,l_death]= extract_tracks(est.X,est.track_list,est.total_tracks);
%%%%%%%%%%
% Plot ground truths
figure;
truths = gcf; hold on;

% Define colors for the tracks
colors = lines(truth.total_tracks);

% Plot tracks in polar coordinates
for i = 1:truth.total_tracks
    if k_birth(i) > 0 && k_death(i) > 0
    Zt = gen_observation_fn(model, X_track(:, k_birth(i):k_death(i), i), 'noiseless');
    
    % Polar coordinates adjustments
    theta = -Zt(1, :) + pi/2; % Adjust angle
    rho = Zt(2, :);          % Range
    
    % Plot track
    if i <= 4
        % Ground truth: Solid line
        h = polar(theta, rho); % Use polar function
        set(h, 'LineWidth', 1.5, 'Color', colors(i, :)); % Modify line properties
    else
        % Ghost track: Dashed line
        h = polar(theta, rho); % Use polar function
        set(h, 'LineWidth', 1.5, 'Color', colors(i, :), 'LineStyle', '--'); % Dashed line for ghost
    end
    
    % Plot start marker
    h_start = polar(theta(1), rho(1), 'ko');
    set(h_start, 'MarkerSize', 8, 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', 'k');
    
    % Plot end marker
    h_end = polar(theta(end), rho(end), 'k^');
    set(h_end, 'MarkerSize', 8, 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', 'k');
    end
end


%%%%%%%%%%%
% Plot ground truths
figure;
truths = gcf; hold on;

% Define colors for the tracks
colors = lines(truth.total_tracks);

% Plot tracks in polar coordinates
for i = 1:truth.total_tracks
    if k_birth(i) > 0 && k_death(i) > 0
    Zt = gen_observation_fn(model, X_track(:, k_birth(i):k_death(i), i), 'noiseless');
    
    % Polar coordinates adjustments
    theta = -Zt(1, :) + pi/2; % Adjust angle
    rho = Zt(2, :);          % Range
    
    % Plot track
    h = polar(theta, rho); % Use polar function
    set(h, 'LineWidth', 1.5, 'Color', colors(i, :)); % Modify line properties
    
    % Plot start marker
    h_start = polar(theta(1), rho(1), 'ko');
    set(h_start, 'MarkerSize', 8, 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', 'k');
    
    % Plot end marker
    h_end = polar(theta(end), rho(end), 'k^');
    set(h_end, 'MarkerSize', 8, 'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', 'k');
    end
end

% Load sensor network data
node_coordinates = define_sensor_locations(method);


adj_matrix = build_adjacency(node_coordinates);

% Plot sensor nodes in Cartesian coordinates
scatter(node_coordinates(:,1), node_coordinates(:,2), 50, 'r', 'filled');

% Plot edges between sensor nodes with updated style
for i = 1:size(adj_matrix, 1)
    for j = i+1:size(adj_matrix, 2)
        if adj_matrix(i, j) == 1
            plot([node_coordinates(i,1), node_coordinates(j,1)], ...
                 [node_coordinates(i,2), node_coordinates(j,2)], 'b--', 'LineWidth', 0.5, 'Color', [0, 0, 1]*0.5);
        end
    end
end

% Customize the figure
title('Ground Truths with Sensor Network', 'FontSize', 14);
axis equal;

% Set the half-disc limits
r_max = 2000; % Radius of the half-disc
xlim([-r_max, r_max]); % Full width for half-disc
ylim([0, r_max]); % Height for half-disc

% Remove grid and node numbers
grid off;

% Save the figure with specified resolution (DPI)
dpi = 300; % Example resolution (DPI)



% Save figure as PNG with the specified DPI
save_filename = fullfile(results_folder, 'ground_truths_with_sensors_no_grid_no_numbers.png');
print(save_filename, '-dpng', ['-r', num2str(dpi)]); % Save at 300 DPI




% Create a structure to store the data
tracking_data = struct();

% Collect x measurements data
x_measurements = {};
for k = 1:meas.K
    if ~isempty(meas.Z{k})
        x_measurements{k} = meas.Z{k}(2,:) .* sin(meas.Z{k}(1,:));
    end   
end
tracking_data.x_measurements = x_measurements;

% Collect x tracks data
x_tracks = {};
for i = 1:truth.total_tracks
    if k_birth(i) > 0 && k_death(i) > 0
        Px = X_track(:, k_birth(i):1:k_death(i), i);
        Px = Px([1 3], :);
        x_tracks{i} = Px;
    end
end
tracking_data.x_tracks = x_tracks;

% Collect x estimates data
x_estimates = [];
for t = 1:size(Y_track, 3)
    x_estimates = [x_estimates; Y_track(1, :, t)];
end
tracking_data.x_estimates = x_estimates;

% Collect y measurements data
y_measurements = {};
for k = 1:meas.K
    if ~isempty(meas.Z{k})
        y_measurements{k} = meas.Z{k}(2,:) .* cos(meas.Z{k}(1,:));
    end
end
tracking_data.y_measurements = y_measurements;

% Collect y tracks data
y_tracks = {};
for i = 1:truth.total_tracks
    if k_birth(i) > 0 && k_death(i) > 0
        Py = X_track(:, k_birth(i):1:k_death(i), i);
        Py = Py([1 3], :);
        y_tracks{i} = Py;
    end
end
tracking_data.y_tracks = y_tracks;

% Collect y estimates data
y_estimates = [];
for t = 1:size(Y_track, 3)
    y_estimates = [y_estimates; Y_track(3, :, t)];
end
tracking_data.y_estimates = y_estimates;

% Save the tracking data to a .mat file
%save('tracking_data.mat', 'tracking_data');
save(fullfile(results_folder, 'tracking_data.mat'), 'tracking_data');

% Continue with your plotting code (already provided in your script)
figure; tracking = gcf; hold on;

% Plot x measurement
subplot(211); box on; 
for k = 1:meas.K
    if ~isempty(meas.Z{k})
        hlined = line(k * ones(size(meas.Z{k}, 2), 1), meas.Z{k}(2,:) .* sin(meas.Z{k}(1,:)), ...
            'LineStyle', 'none', 'Marker', 'x', 'Markersize', 6, 'Color', 0.7 * ones(1, 3)); % Marker size set to 6
    end
end

% Plot x track
for i = 1:truth.total_tracks
    if k_birth(i) > 0 && k_death(i) > 0
        Px = X_track(:, k_birth(i):1:k_death(i), i); Px = Px([1 3], :);
        hline1 = line(k_birth(i):1:k_death(i), Px(1,:), 'LineStyle', '-', 'Marker', 'none', ...
            'LineWidth', 1, 'Color', 0 * ones(1, 3)); % True track line (consistent color)
    end
end

% Plot x estimate
for t = 1:size(Y_track, 3)
    hline2 = line(1:truth.K, Y_track(1, :, t), 'LineStyle', 'none', 'Marker', 'o', 'Markersize', 6, ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', colorarray.rgb(t,:)); % Circle marker with black border and size 6
end

% Plot y measurement
subplot(212); box on;
for k = 1:meas.K
    if ~isempty(meas.Z{k})
        yhlined = line(k * ones(size(meas.Z{k}, 2), 1), meas.Z{k}(2,:) .* cos(meas.Z{k}(1,:)), ...
            'LineStyle', 'none', 'Marker', 'x', 'Markersize', 6, 'Color', 0.7 * ones(1, 3)); % Marker size set to 6
    end
end

% Plot y track
for i = 1:truth.total_tracks
    if k_birth(i) > 0 && k_death(i) > 0
        Py = X_track(:, k_birth(i):1:k_death(i), i); Py = Py([1 3], :);
        yhline1 = line(k_birth(i):1:k_death(i), Py(2,:), 'LineStyle', '-', 'Marker', 'none', ...
            'LineWidth', 1, 'Color', 0 * ones(1, 3)); % True track line (consistent color)
    end
end

% Plot y estimate
for t = 1:size(Y_track, 3)
    hline2 = line(1:truth.K, Y_track(3, :, t), 'LineStyle', 'none', 'Marker', 'o', 'Markersize', 6, ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', colorarray.rgb(t,:)); % Circle marker with black border and size 6
end



% if the loop never ran, make a dummy handle
if ~exist('hline2','var') || isempty(hline2)
    % pick a safe fallback face color:
    if isfield(colorarray,'rgb') && size(colorarray.rgb,1) >= 1
        faceColor = colorarray.rgb(1,:);
    else
        faceColor = [0 0 0];           % default to black
        % or: faceColor = 'none';     % no fill at all
    end
    
    hline2 = line(NaN, NaN, ...
        'LineStyle',     'none', ...
        'Marker',        'o', ...
        'MarkerSize',    6, ...
        'MarkerEdgeColor','k', ...
        'MarkerFaceColor', faceColor, ...
        'Visible',       'off');       % stays hidden
end

% Add labels and save the figure
set(gcf, 'DefaultAxesFontName', 'Times New Roman', 'DefaultTextFontName', 'Times New Roman', 'DefaultAxesFontSize', 12, 'DefaultLegendFontSize', 12);

% Set Times New Roman font and decrease axis numbers font size slightly
subplot(211);  
xlabel('Time', 'FontSize', 14); ylabel('x-coordinate (m)', 'FontSize', 14);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12); % Slightly decreased font size for axis numbers
set(gca, 'XLim', [1 truth.K]); set(gca, 'YLim', [-model.range_c(2, 2) model.range_c(2, 2)]);
legend([hline2 hline1 hlined], 'Estimates          ', 'True tracks', 'Measurements','FontSize', 12);

subplot(212); 
xlabel('Time','FontSize', 14); ylabel('y-coordinate (m)','FontSize', 14);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12); % Slightly decreased font size for axis numbers
set(gca, 'XLim', [1 truth.K]); set(gca, 'YLim', [model.range_c(1, 2) model.range_c(2, 2)]);

% Save the restored figure
saveas(gcf, fullfile(results_folder, 'tracking.png')); % Save figure


%plot error
ospa_vals= zeros(truth.K,3);
ospa_c= 100;
ospa_p= 1;
for k=1:meas.K
    [ospa_vals(k,1), ospa_vals(k,2), ospa_vals(k,3)]= ospa_dist(get_comps(truth.X{k},[1 3]),get_comps(est.X{k},[1 3]),ospa_c,ospa_p);
end

figure; ospa = gcf; hold on;

% Create a data structure to store OSPA values
ospa_data = struct('time', 1:meas.K, 'dist', [], 'loc', [], 'card', []);

% Plot OSPA Dist
subplot(3,1,1); 
plot(1:meas.K, ospa_vals(:,1), 'k'); 
grid on; 
set(gca, 'XLim', [1 meas.K], 'YLim', [0 ospa_c]); 
ylabel('OSPA Dist');
% Store OSPA Dist in data structure
ospa_data.dist = ospa_vals(:,1);

% Plot OSPA Loc
subplot(3,1,2); 
plot(1:meas.K, ospa_vals(:,2), 'k'); 
grid on; 
set(gca, 'XLim', [1 meas.K], 'YLim', [0 ospa_c]); 
ylabel('OSPA Loc');
% Store OSPA Loc in data structure
ospa_data.loc = ospa_vals(:,2);

% Plot OSPA Card
subplot(3,1,3); 
plot(1:meas.K, ospa_vals(:,3), 'k'); 
grid on; 
set(gca, 'XLim', [1 meas.K], 'YLim', [0 ospa_c]); 
ylabel('OSPA Card');
% Store OSPA Card in data structure
ospa_data.card = ospa_vals(:,3);

xlabel('Time');

% Save the figure
saveas(gcf, fullfile(results_folder, 'ospa.png')); 

% Save the data
save(fullfile(results_folder, 'ospa_data.mat'), 'ospa_data');



% Calculate mean values for each component of OSPA
ospa_dist_mean = mean(ospa_vals(:,1));
ospa_loc_mean = mean(ospa_vals(:,2));
ospa_card_mean = mean(ospa_vals(:,3));

% Create a table to store the mean values
mean_values = table(ospa_dist_mean, ospa_loc_mean, ospa_card_mean, ...
                    'VariableNames', {'OSPA_Dist_Mean', 'OSPA_Loc_Mean', 'OSPA_Card_Mean'});

% Save the table as a CSV file
csv_file_path = fullfile(results_folder, 'OSPA_Mean_Values.csv');
writetable(mean_values, csv_file_path);




%plot error - OSPA^(2)
order = 1;
cutoff = 100;
win_len= 10;

ospa2_cell = cell(1,length(win_len));
for i = 1:length(win_len)
    ospa2_cell{i} = compute_ospa2(X_track([1 3],:,:),Y_track([1 3],:,:),cutoff,order,win_len);
end

% Initialize the mean OSPA^2 data storage
mean_ospa2_dist = zeros(1, length(win_len));
mean_ospa2_loc = zeros(1, length(win_len));
mean_ospa2_card = zeros(1, length(win_len));

% Compute the mean values for each metric
for i = 1:length(win_len)
    mean_ospa2_dist(i) = mean(ospa2_cell{i}(1, :));
    mean_ospa2_loc(i) = mean(ospa2_cell{i}(2, :));
    mean_ospa2_card(i) = mean(ospa2_cell{i}(3, :));
end

% Prepare data for CSV
mean_ospa2_table = table(win_len', mean_ospa2_dist', mean_ospa2_loc', mean_ospa2_card', ...
    'VariableNames', {'Window_Length', 'Mean_OSPA2_Dist', 'Mean_OSPA2_Loc', 'Mean_OSPA2_Card'});

% Save the mean OSPA^2 values to a CSV file
csv_filename = fullfile(results_folder, 'mean_ospa2_values.csv');
writetable(mean_ospa2_table, csv_filename);

disp(['Mean OSPA^2 values saved to: ', csv_filename]);


figure; ospa2 = gcf; hold on;

% Initialize labels and data structure
windowlengthlabels = cell(1, length(win_len));
ospa2_data = struct('time', 1:truth.K, 'ospa_dist', [], 'ospa_loc', [], 'ospa_card', []);

subplot(3,1,1);
for i = 1:length(win_len)
    plot(1:truth.K, ospa2_cell{i}(1,:), 'k'); 
    grid on; 
    set(gca, 'XLim', [1 meas.K]); 
    set(gca, 'YLim', [0 cutoff]); 
    ylabel('OSPA$^{(2)}$ Dist', 'interpreter', 'latex');
    windowlengthlabels{i} = ['$L_w = ' int2str(win_len(i)) '$'];
    % Store OSPA distance data
    ospa2_data.ospa_dist(i, :) = ospa2_cell{i}(1,:);
end
legend(windowlengthlabels, 'interpreter', 'latex');

subplot(3,1,2);
for i = 1:length(win_len)
    plot(1:truth.K, ospa2_cell{i}(2,:), 'k'); 
    grid on; 
    set(gca, 'XLim', [1 meas.K]); 
    set(gca, 'YLim', [0 cutoff]); 
    ylabel('OSPA$^{(2)}$ Loc', 'interpreter', 'latex');
    % Store OSPA location data
    ospa2_data.ospa_loc(i, :) = ospa2_cell{i}(2,:);
end

subplot(3,1,3);
for i = 1:length(win_len)
    plot(1:truth.K, ospa2_cell{i}(3,:), 'k'); 
    grid on; 
    set(gca, 'XLim', [1 meas.K]); 
    set(gca, 'YLim', [0 cutoff]); 
    ylabel('OSPA$^{(2)}$ Card', 'interpreter', 'latex');
    % Store OSPA cardinality data
    ospa2_data.ospa_card(i, :) = ospa2_cell{i}(3,:);
end
xlabel('Time', 'interpreter', 'latex');

% Save the figure
saveas(gcf, fullfile(results_folder, 'ospa2.png')); 

% Save the data
save(fullfile(results_folder, 'ospa2_data.mat'), 'ospa2_data');

   

figure; cardinality = gcf; 
subplot(2,1,1); box on; hold on;

% Plot the true and estimated cardinalities
stairs(1:meas.K, truth.N, 'k', 'LineWidth', 1.5); % True cardinality as solid line
plot(1:meas.K, est.N, 'k--', 'LineWidth', 1.5);   % Estimated cardinality as dashed line

% Customize grid and legend
grid on;
set(gca, 'GridLineStyle', '--', 'GridColor', [0.6, 0.6, 0.6], 'GridAlpha', 0.5); % Light dashed grid
legend(gca, 'True', 'Estimated', 'Location', 'Best');

% Set axis limits and labels
set(gca, 'XLim', [1 meas.K], 'YLim', [0 20]);
xlabel('Time'); 
ylabel('Cardinality');
title('Cardinality Over Time', 'FontWeight', 'Bold');

% Save the plot data
plot_data.time = 1:meas.K;
plot_data.truth = truth.N;
plot_data.estimated = est.N;
save(fullfile(results_folder, 'plot_card.mat'), 'plot_data');
%save('plot_data.mat', 'plot_data'); % Save data into a .mat file


%return
handles=[ truths tracking ospa ospa2 cardinality ];
  saveas(gcf, fullfile(results_folder, 'cardinality.png')); % Save figure


function ca= makecolorarray(nlabels)
    lower= 0.1;
    upper= 0.9;
    rrr= rand(1,nlabels)*(upper-lower)+lower;
    ggg= rand(1,nlabels)*(upper-lower)+lower;
    bbb= rand(1,nlabels)*(upper-lower)+lower;
    ca.rgb= [rrr; ggg; bbb]';
    ca.lab= cell(nlabels,1);
    ca.cnt= 0;   
end

function idx= assigncolor(label)
    str= sprintf('%i*',label);
    tmp= strcmp(str,colorarray.lab);
    if any(tmp)
        idx= find(tmp);
    else
        colorarray.cnt= colorarray.cnt + 1;
        colorarray.lab{colorarray.cnt}= str;
        idx= colorarray.cnt;
    end
end

function count= countestlabels
    labelstack= [];
    for k=1:meas.K
        labelstack= [labelstack est.L{k}];
    end
    [c,~,~]= unique(labelstack','rows');
    count=size(c,1);
end
close all; % Close all figure windows after saving them

end


function [X_track,k_birth,k_death]= extract_tracks(X,track_list,total_tracks)

K= size(X,1); 
x_dim= size(X{K},1); 
k=K-1; while x_dim==0, x_dim= size(X{k},1); k= k-1; end
X_track= NaN(x_dim,K,total_tracks);
k_birth= zeros(total_tracks,1);
k_death= zeros(total_tracks,1);

max_idx= 0;
for k=1:K
    if ~isempty(X{k})
        X_track(:,k,track_list{k})= X{k};
    end
    if max(track_list{k})> max_idx %new target born?
        idx= find(track_list{k}> max_idx);
        k_birth(track_list{k}(idx))= k;
    end
    if ~isempty(track_list{k}), max_idx= max(track_list{k}); end
    k_death(track_list{k})= k;
end
end


function Xc= get_comps(X,c)

if isempty(X)
    Xc= [];
else
    Xc= X(c,:);
end
end
