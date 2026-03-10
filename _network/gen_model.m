function model= gen_model(x,y,num_nodes)
model.num_nodes=num_nodes;
%sensor location
model.x_off=x;
model.y_off=y;
% Define FoV parameters
model.FoV_r_max = 20000;                  % Maximum range in meters
% basic parameters
model.x_dim= 5;   %dimension of state vector
model.z_dim= 2;   %dimension of observation vector
model.v_dim= 3;   %dimension of process noise
model.w_dim= 2;   %dimension of observation noise

% dynamical model parameters (CT model)
% state transformation given by gen_newstate_fn, transition matrix is N/A in non-linear case
model.T= 1;                         %sampling period
model.sigma_vel= 5;
model.sigma_turn= (pi/180);   %std. of turn rate variation (rad/s)
model.bt= model.sigma_vel*[ (model.T^2)/2; model.T ];
model.B2= [ model.bt zeros(2,2); zeros(2,1) model.bt zeros(2,1); zeros(1,2) model.T*model.sigma_turn ];
model.B= eye(model.v_dim);
model.Q= model.B*model.B';

% survival/death parameters
model.P_S= .99;
model.Q_S= 1-model.P_S;

% birth parameters (LMB birth model, single component only)
model.T_birth= 4;         %no. of LMB birth terms
model.L_birth= zeros(model.T_birth,1);                                          %no of Gaussians in each LMB birth term
model.r_birth= zeros(model.T_birth,1);                                          %prob of birth for each LMB birth term
model.w_birth= cell(model.T_birth,1);                                           %weights of GM for each LMB birth term
model.m_birth= cell(model.T_birth,1);                                           %means of GM for each LMB birth term
model.B_birth= cell(model.T_birth,1);                                           %std of GM for each LMB birth term
model.P_birth= cell(model.T_birth,1);                                           %cov of GM for each LMB birth term

model.L_birth(1)=1;                                                             %no of Gaussians in birth term 1
model.r_birth(1)=0.02;                                                          %prob of birth
model.w_birth{1}(1,1)= 1;                                                       %weight of Gaussians - must be column_vector
model.m_birth{1}(:,1)= [ -1500; 0; 250; 0; 0 ];                                 %mean of Gaussians
model.B_birth{1}(:,:,1)= diag([ 50; 50; 50; 50; 6*(pi/180) ]);                  %std of Gaussians
model.P_birth{1}(:,:,1)= model.B_birth{1}(:,:,1)*model.B_birth{1}(:,:,1)';      %cov of Gaussians

model.L_birth(2)=1;                                                             %no of Gaussians in birth term 2
model.r_birth(2)=0.02;                                                          %prob of birth
model.w_birth{2}(1,1)= 1;                                                       %weight of Gaussians - must be column_vector
model.m_birth{2}(:,1)= [ -250; 0; 1000; 0; 0 ];                                 %mean of Gaussians
model.B_birth{2}(:,:,1)= diag([ 50; 50; 50; 50; 6*(pi/180) ]);                  %std of Gaussians
model.P_birth{2}(:,:,1)= model.B_birth{1}(:,:,1)*model.B_birth{1}(:,:,1)';      %cov of Gaussians

model.L_birth(3)=1;                                                             %no of Gaussians in birth term 3
model.r_birth(3)=0.03;                                                          %prob of birth
model.w_birth{3}(1,1)= 1;                                                       %weight of Gaussians - must be column_vector
model.m_birth{3}(:,1)= [ 250; 0; 750; 0; 0 ];                                   %mean of Gaussians
model.B_birth{3}(:,:,1)= diag([ 50; 50; 50; 50; 6*(pi/180) ]);                  %std of Gaussians
model.P_birth{3}(:,:,1)= model.B_birth{1}(:,:,1)*model.B_birth{1}(:,:,1)';      %cov of Gaussians

model.L_birth(4)=1;                                                             %no of Gaussians in birth term 4
model.r_birth(4)=0.03;                                                          %prob of birth
model.w_birth{4}(1,1)= 1;                                                       %weight of Gaussians - must be column_vector
model.m_birth{4}(:,1)= [ 1000; 0; 1500; 0; 0 ];                                 %mean of Gaussians
model.B_birth{4}(:,:,1)= diag([ 50; 50; 50; 50; 6*(pi/180) ]);                  %std of Gaussians
model.P_birth{4}(:,:,1)= model.B_birth{1}(:,:,1)*model.B_birth{1}(:,:,1)';      %cov of Gaussians

% observation model parameters (noisy r/theta only)
% measurement transformation given by gen_observation_fn, observation matrix is N/A in non-linear case
model.D= diag([ 2*(pi/180); 10 ]);      %std for angle and range noise
model.R= model.D*model.D';              %covariance for observation noise

% detection parameters
model.P_D= .98;   %probability of detection in measurements
model.Q_D= 1-model.P_D; %probability of missed detection in measurements

% clutter parameters
model.lambda_c= 10;                             %poisson average rate of uniform clutter (per scan)
model.range_c= [ -pi/2 pi/2; 0 2000 ];          %uniform clutter on r/theta
model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); %uniform clutter density



% ============================================================
% Detection probability and clutter rate sensitivity switch
% ============================================================
% sw = 1 : benign
% sw = 2 : moderate clutter / moderate PD loss
% sw = 3 : high clutter / low PD  (benchmark case)
% sw = 4 : extreme stress
% ============================================================

sw = 1;   % <<< SET THIS SWITCH >>> (1, 2, 3, or 4)

switch sw
    case 1
        model.P_D     = 0.99;
        model.lambda_c = 0;

    case 2
        model.P_D     = 0.80;
        model.lambda_c = 30;

    case 3
        model.P_D     = 0.70;
        model.lambda_c = 60;   % benchmark stress case

    case 4
        model.P_D     = 0.60;
        model.lambda_c = 90;

    otherwise
        error('Invalid switch value: sw must be 1, 2, 3, or 4');
end

% Missed detection probability
model.Q_D = 1 - model.P_D;

% ============================================================
% Clutter spatial model (unchanged across experiments)
% ============================================================
model.range_c = [ -pi/2  pi/2; 
                   0     2000 ];

model.pdf_c = 1 / prod(model.range_c(:,2) - model.range_c(:,1));


model.Tn=1;
% sw = 5;   % <<< SET THIS SWITCH >>> (0, 1, 2, 3,4, 5) 
% 
% switch sw
%     case 0
%         model.Tn=0;
%     case 1
%         model.Tn=1;
% 
%     case 2
%         model.Tn=2;
% 
%     case 3
%        model.Tn=3;
% 
%     case 4
%         model.Tn=4;
%     case 5
%         model.Tn=5;
% 
%     otherwise
%         error('Invalid switch value: sw must be 1, 2, 3, or 4');
% end




% % ============================================================
% % Measurement noise sensitivity switch (angle/range)
% % Observation model: z = [theta; r] with additive noise
% % model.D stores standard deviations (std) for [theta; r]
% % model.R is the corresponding covariance
% % ============================================================
% 
% swR = 3;   % <<< SET THIS SWITCH >>> (1..5)
% 
% % Nominal (your current setting)
% sigma_theta_nom = 2*(pi/180);   % rad
% sigma_r_nom     = 10;           % m
% 
% switch swR
%     case 1
%         % Very low noise (optimistic sensor)
%         sigma_theta = 0.5 * sigma_theta_nom;
%         sigma_r     = 0.5 * sigma_r_nom;
% 
%     case 2
%         % Low noise
%         sigma_theta = 0.75 * sigma_theta_nom;
%         sigma_r     = 0.75 * sigma_r_nom;
% 
%     case 3
%         % Nominal (baseline)
%         sigma_theta = 1.0 * sigma_theta_nom;
%         sigma_r     = 1.0 * sigma_r_nom;
% 
%     case 4
%         % High noise
%         sigma_theta = 1.5 * sigma_theta_nom;
%         sigma_r     = 1.5 * sigma_r_nom;
% 
%     case 5
%         % Very high noise (pessimistic sensor)
%         sigma_theta = 2.0 * sigma_theta_nom;
%         sigma_r     = 2.0 * sigma_r_nom;
% 
%     otherwise
%         error('Invalid swR: must be an integer in {1,2,3,4,5}');
% end
% 
% % Build noise model
% model.D = diag([sigma_theta; sigma_r]);  % std for angle and range noise
% model.R = model.D * model.D';            % covariance for observation noise
% 
% % Optional trace log (recommended for reproducibility)
% fprintf('[R-sens] swR=%d | sigma_theta=%.4g rad | sigma_r=%.4g m\n', ...
%         swR, sigma_theta, sigma_r);



% 
% % ============================================================
% % Measurement noise sensitivity (used by the FILTER)
% % Observation: z = [theta; r]
% % ============================================================
% 
% swR = 4;   % <<< SET THIS SWITCH >>> (1..5)
% 
% % Nominal noise (baseline in the paper)
% sigma_theta_nom = 2*(pi/180);   % rad
% sigma_r_nom     = 10;           % m
% 
% switch swR
%     case 1
%         % Very low noise (optimistic sensor / strong replay consistency)
%         sigma_theta = 0.5 * sigma_theta_nom;
%         sigma_r     = 0.5 * sigma_r_nom;
% 
%     case 2
%         % Low noise
%         sigma_theta = 0.75 * sigma_theta_nom;
%         sigma_r     = 0.75 * sigma_r_nom;
% 
%     case 3
%         % Nominal
%         sigma_theta = 1.0 * sigma_theta_nom;
%         sigma_r     = 1.0 * sigma_r_nom;
% 
%     case 4
%         % High noise
%         sigma_theta = 1.5 * sigma_theta_nom;
%         sigma_r     = 1.5 * sigma_r_nom;
% 
%     case 5
%         % Very high noise (pessimistic sensor)
%         sigma_theta = 2.0 * sigma_theta_nom;
%         sigma_r     = 2.0 * sigma_r_nom;
% 
%     otherwise
%         error('Invalid swR: must be in {1,2,3,4,5}');
% end
% 
% % ------------------------------------------------------------
% % Final measurement noise model USED BY THE FILTER
% % ------------------------------------------------------------
% model.D = diag([sigma_theta; sigma_r]);   % std deviations
% model.R = model.D * model.D';             % covariance matrix

% % ------------------------------------------------------------
% % Optional logging (recommended for reproducibility)
% % ------------------------------------------------------------
% fprintf(['[Meas-noise] swR=%d | sigma_theta=%.2f deg | ' ...
%          'sigma_r=%.2f m\n'], ...
%          swR, sigma_theta*180/pi, sigma_r);
