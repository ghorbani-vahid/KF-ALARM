function meas= gen_meas(model,truth, attack)
rng(42);
%variables
meas.K= truth.K;
meas.Z= cell(truth.K,1);

%generate measurements
for k=1:truth.K
    if truth.N(k) > 0
        idx= find( rand(truth.N(k),1) <= model.P_D  );         %compute_pD(model,truth.X{k})                %detected target indices
        if k > 4
            idx= find( rand(truth.N(k),1) <= model.P_D ) ;                                                % ghost measurements
        end    
        meas.Z{k}= gen_observation_fn(model,truth.X{k}(:,idx),'noise');                          %single target observations if detected 
        meas.ZZ{k}= gen_observation_fn(model,truth.X{k}(:,idx),'noiseless');   
        if  k>10  && strcmp(attack.scenario, 'none-for-replay')                                                %if attack.scenario=="replay" && k>attack.delay   % set attack scenario to none to get replay scenario now
          meas.Z{k}= [meas.Z{k} meas.ZZ{k-10}];               % inject old scan
        end
    end
    N_c= poissrnd(model.lambda_c);                                                               %number of clutter points
    C= repmat(model.range_c(:,1),[1 N_c])+ diag(model.range_c*[ -1; 1 ])*rand(model.z_dim,N_c);  %clutter generation
    meas.Z{k}= [ meas.Z{k} C ];                                                                  %measurement is union of detections and clutter
end
    