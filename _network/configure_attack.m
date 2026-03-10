function [attacked_sensor, attack_scenario, ghost_num] = configure_attack(method)
    attack_scenario = input('Ghost attack scenario (deception, replay, flooding, none): ', 's');
     if strcmpi(attack_scenario, 'deception')
        ghost_num = input('Number of ghosts (4, 6, 12, 16): ');
    else
        ghost_num = 0;  
    end
    if strcmp(method, 'ST')
        attacked_sensor = 1;
    else
        attacked_sensor = [2];
    end
end