function node_struct_array = convertToStructArray(nn)
    node_struct_array = struct('id',[],'x_position',[],'y_position',[],...
        'model',[],'truth',[],'meas',[],'est_current',[],'neighbors',[]);
    for k = 1:numel(nn)
        node_struct_array(k).id          = nn{k}.id;
        node_struct_array(k).x_position  = nn{k}.x_position;
        node_struct_array(k).y_position  = nn{k}.y_position;
        node_struct_array(k).model       = nn{k}.model;
        node_struct_array(k).truth       = nn{k}.truth;
        node_struct_array(k).meas        = nn{k}.meas;
        node_struct_array(k).est_current = nn{k}.est_current;
        node_struct_array(k).neighbors   = nn{k}.neighbors;
    end
end