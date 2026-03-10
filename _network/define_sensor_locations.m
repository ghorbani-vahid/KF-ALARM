function sensor_locations = define_sensor_locations(method)
    if strcmp(method, 'ALARM')
        sensor_locations = [
            -1500, 1.84E-13;
            -935.2347028, 570;
             935.2347028, 570;
            -2000, 0;
            -1618.033989, 1175.570505;
            -618.0339887, 1902.113033;
             618.0339887, 1902.113033;
            1618.033989, 1175.570505;
            2000, 0;
            1500, 1.84E-13;
        ];
    elseif strcmp(method, 'ST')
        sensor_locations = [-935.2347028, 570];
    else
        error('Unknown filtering method. Please enter "ALARM" or "ST".');
    end
end