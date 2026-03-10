function est = run_node(model,meas,truth, k,tt,tt_lmb_update_pre, nn,iter_num, estt,id, save,x_off, y_off, method) %#ok<INUSD>
rng(42);

% =========================================================================
% PIPELINE-COMPATIBLE INTERFACE
% =========================================================================
% This function implements a multi-target Extended Kalman Filter (EKF)
% tracker with coordinated-turn (CT) dynamics and polar measurements.
%
% IMPORTANT IMPLEMENTATION NOTE
% -------------------------------------------------------------------------
% Although several variable names in this file retain LMB/GLMB-style
% notation (for example: tt_lmb_update, clean_lmb, extract_estimates_score),
% the underlying filtering logic in this implementation is NOT a formal
% Labeled Multi-Bernoulli (LMB) or Generalized Labeled Multi-Bernoulli
% (GLMB) recursion.
%
% These names are preserved only for compatibility with an existing
% research pipeline that originally used LMB/GLMB-style interfaces.
% Renaming them would break compatibility with the surrounding codebase.
%
% In particular:
%   - Each "track" here is an EKF-maintained Gaussian state estimate.
%   - trk.r is a heuristic confidence/track score, NOT a formal Bernoulli
%     existence probability in the strict RFS sense.
%   - trk.l is retained as metadata / pipeline bookkeeping, NOT as a strict
%     labeled-RFS object identity with full GLMB/LMB semantics.
%   - Birth, pruning, and extraction steps are score-based engineering
%     mechanisms designed to mimic the surrounding pipeline structure, not
%     exact LMB/GLMB posterior propagation.
%
% Therefore, this code should be interpreted as:
%   "An EKF-based multi-target tracker implemented inside a legacy
%    LMB/GLMB-compatible interface."
%
% TRACK FIELDS
% -------------------------------------------------------------------------
%   trk.x          : 5x1 EKF state mean
%   trk.w          : 5x5 EKF state covariance
%   trk.r          : confidence score for track management
%   trk.l          : legacy metadata / pipeline-compatible label field
%   trk.age        : track age in scans
%   trk.hit_streak : recent successful update count
%   trk.miss_streak: consecutive miss count
%   trk.maturity   : long-term persistence memory
%   trk.prev_pos   : previous position [px; py]
%   trk.prev_vel   : previous velocity [vx; vy]
%
% =========================================================================
% THIS VERSION
% =========================================================================
% 1) PRIOR-BIRTH MODEL (LMB-LIKE INTERFACE ONLY):
%    - weak prior birth tracks are created each scan from model birth terms
%    - no direct measurement-triggered spawning is used here
%
% 2) SINGLE-HYPOTHESIS EKF TRACKING:
%    - each maintained track is propagated by EKF predict/update
%    - measurement-to-track assignment is solved globally by Hungarian
%
% 3) PIPELINE-COMPATIBLE SCORE MANAGEMENT:
%    - track confidence is updated by hit/miss logic
%    - pruning and estimate extraction use score/maturity thresholds
%
% 4) NEIGHBOR-CORROBORATION / ALARM-LIKE LOGIC:
%    - neighboring nodes are used only to compute a reliability factor
%    - this reliability modulates measurement trust
%    - this is not an LMB/GLMB multi-object Bayes recursion
%
% 5) HARD STALE-TRACK PRUNING:
%    - stale tracks are removed before and after update
%    - this prevents weak zombie tracks from lingering/reappearing
% =========================================================================
%=== outputs
est.X= cell(meas.K,1);
est.N= zeros(meas.K,1);
est.L= cell(meas.K,1);

%=== filter parameters
filter.T_max= 200;
filter.track_threshold= 0.05;
filter.run_flag= 'disp';

% gating
filter.P_G= 0.999;
filter.gate_flag= 1;
filter.gate_chi2 = chi2inv_safe(filter.P_G, model.z_dim);
filter.max_innov_lik_floor = 1e-300;

% ------------------- PRIOR BIRTH MODEL -----------------------------------
filter.birth_r0 = 0.02;     % weak prior birth existence/score
filter.birth_age0 = 1;
filter.birth_hit0 = 0;
filter.birth_miss0 = 0;
filter.birth_maturity0 = 0;
filter.label_base = 100000;
% -------------------------------------------------------------------------

% estimation
filter.est_r_min = 0.30;
filter.est_maxN  = 50;
filter.est_min_maturity = 2.0;

% ------------------- EKF-ALARM knobs -------------------------------------
filter.alarm_eps_alpha = 0.05;
filter.alarm_pos_epsP  = 1e-6;
filter.alarm_c_null    = 1e6;
filter.alarm_cost_gate = 80;
filter.alarm_count_only_nonempty = true;
filter.alarm_hit_penalty = true;
filter.alarm_miss_penalty = true;
% -------------------------------------------------------------------------

% ------------------- HUNGARIAN ASSOCIATION KNOBS -------------------------
filter.assign_unmatched_loglik = log(1e-12);
filter.assign_infeasible_cost = 1e9;
% -------------------------------------------------------------------------

% ------------------- CONTINUITY / AGE / MATURITY -------------------------
filter.cont_use = true;
filter.cont_sigma_pos = 120;        % meters
filter.cont_lambda = 3.5;
filter.cont_max_penalty = 12.0;

% very mild direction term
filter.dir_use = true;
filter.dir_lambda = 0.5;
filter.dir_cos_bad = -0.20;
filter.dir_penalty_bad = 1.0;
filter.dir_penalty_side = 0.3;
filter.dir_penalty_stop = 0.05;
filter.dir_speed_floor = 0.75;

filter.age_use = true;
filter.age_confirm_hits = 5;
filter.age_bonus_lambda = 1.2;
filter.age_bonus_cap = 2.0;

filter.maturity_use = true;
filter.maturity_hit_inc = 1.0;
filter.maturity_miss_decay = 0.35;
filter.maturity_bonus_lambda = 1.6;
filter.maturity_bonus_cap = 2.5;
filter.maturity_confirm = 5.0;

filter.protect_confirmed = true;
filter.protect_min_hits = 5;
filter.protect_min_age  = 7;
filter.protect_min_maturity = 5.0;
filter.protect_bonus_extra = 1.0;
% -------------------------------------------------------------------------

% ------------------- GENTLE ANTI-SWITCH BIAS -----------------------------
filter.weak_track_cost = 0.40;
filter.soft_reserve_use = true;
filter.soft_reserve_R = 140;
filter.soft_reserve_max_miss = 2;
filter.soft_reserve_penalty = 0.75;
% -------------------------------------------------------------------------

% ------------------- HARD STALE-TRACK PRUNING ----------------------------
filter.prune_use = true;
filter.prune_r_hard = 0.09;             % hard delete if below this
filter.prune_birth_hit_max = 1;         % birth-like if hit_streak <= this
filter.prune_birth_maturity_max = 1.0;  % and maturity <= this
filter.prune_birth_miss_max = 1;        % kill birth-like tracks quickly

filter.prune_weak_miss_max = 0;         % weak tracks: short leash
filter.prune_mature_miss_max = 1;       % mature tracks: more tolerance

filter.prune_very_old_weak_age = 12;    % weak track lingering too long
filter.prune_very_old_weak_r = 0.20;     % if old and still weak, prune
% -------------------------------------------------------------------------

est.filter = filter;

% NOTE:
% The variable name "tt_lmb_update" is kept only for interface consistency
% with the original pipeline. In this implementation it simply stores the
% current list of EKF-managed tracks; it is not an LMB posterior.
if k==1
    tt_lmb_update = cell(0,1);
else
    tt_lmb_update = tt_lmb_update_pre;
end

% predict + update
[tt_lmb_update, diaginfo] = ...
    jointlmbpredictupdate(tt_lmb_update, model, filter, meas, k, tt);

T_predict   = diaginfo.T_predict;
T_posterior = diaginfo.T_posterior;

% prune/cap
tt_lmb_update = clean_lmb(tt_lmb_update, filter);
T_clean = length(tt_lmb_update);

% estimates
[estt.X{k, 1}, estt.N(k, 1), estt.L{k, 1}] = extract_estimates_score(tt_lmb_update, model, filter);
est = estt;

% diag
display_diaginfo(tt_lmb_update, k, est, filter, T_predict, T_posterior, T_clean);

est.tt_lmb_update = tt_lmb_update;

if k==iter_num
    if save=="true"
        attack.scenario="none";
        attack.intensity=0;
        truth = gen_truth(model, iter_num, attack);
        plot_results(model, truth, meas, estt, id, method);
    end
end
end

% NOTE:
% Despite the legacy function name, this routine performs EKF-based track
% prediction/update plus score-based track management inside a pipeline that
% originally expected LMB-style objects.
% =========================================================================
function [tt_out, diaginfo] = jointlmbpredictupdate(tt_lmb_update, model, filter, meas, k, tt)

% -------------------------------------------------------------------------
% LEGACY NAME WARNING
% -------------------------------------------------------------------------
% This function name is retained from the original LMB-oriented pipeline.
% However, the operations below are EKF-based single-track prediction and
% update steps, combined with score-based birth/survival/pruning logic.
% No formal LMB/GLMB posterior recursion is carried out here.


% -------------------------------------------------------------------------
% HARD PRUNE STALE TRACKS BEFORE PREDICTION
% -------------------------------------------------------------------------
tt_lmb_update = prune_dead_tracks_hard(tt_lmb_update, filter);

% -------------------------------------------------------------------------
% PRIOR BIRTHS (LMB-LIKE): weak, measurement-independent birth hypotheses
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% PRIOR BIRTH GENERATION
% -------------------------------------------------------------------------
% These "births" are engineering prior track hypotheses inserted for
% pipeline consistency. They should not be interpreted as a rigorous
% Bernoulli birth model unless the surrounding mathematical formulation
% explicitly justifies that interpretation.

tt_birth = make_prior_births(model, filter, k);

% --- predict surviving tracks
Qk = model.B2 * model.Q * model.B2.';
tt_survive = cell(length(tt_lmb_update),1);
for i = 1:length(tt_lmb_update)
    trk = tt_lmb_update{i};

    prev_pos = trk.x([1;3]);
    prev_vel = trk.x([2;4]);

    r_pred = model.P_S * trk.r;
    [m_pred, P_pred] = ekf_predict_ct(model, trk.x, trk.w, Qk);

    tt_survive{i}.r = r_pred;
    tt_survive{i}.x = m_pred;
    tt_survive{i}.w = P_pred;
    tt_survive{i}.l = trk.l; %#ok<NASGU>
    tt_survive{i}.age = getfield_default(trk,'age',1) + 1;
    tt_survive{i}.hit_streak = getfield_default(trk,'hit_streak',0);
    tt_survive{i}.miss_streak = getfield_default(trk,'miss_streak',0);
    tt_survive{i}.maturity = getfield_default(trk,'maturity',0);
    tt_survive{i}.prev_pos = prev_pos;
    tt_survive{i}.prev_vel = prev_vel;
end

tt_predict = cat(1, tt_survive, tt_birth);

diaginfo.T_predict   = length(tt_predict);
diaginfo.T_posterior = length(tt_predict);

% --- measurements
Z = meas.Z{k};
m = size(Z,2);
nT = length(tt_predict);

if nT==0
    tt_out = tt_predict;
    return;
end

if m==0
    for i=1:nT
        tt_predict{i}.r = score_missed(tt_predict{i}.r, model.P_D);

        miss0 = getfield_default(tt_predict{i},'miss_streak',0);
        hit0  = getfield_default(tt_predict{i},'hit_streak',0);
        mat0  = getfield_default(tt_predict{i},'maturity',0);

        tt_predict{i}.miss_streak = miss0 + 1;
        tt_predict{i}.hit_streak  = max(hit0 - 1, 0);
        tt_predict{i}.maturity    = max(mat0 - filter.maturity_miss_decay, 0);
    end

    tt_out = prune_dead_tracks_hard(tt_predict, filter);
    diaginfo.T_posterior = length(tt_out);
    return;
end

% =========================================================================
% EKF-ALARM: compute track reliability alpha_i(t) from neighbor corroboration
% =========================================================================
% -------------------------------------------------------------------------
% NEIGHBOR CORROBORATION / RELIABILITY
% -------------------------------------------------------------------------
% This block compares locally predicted EKF tracks with neighboring node
% tracks using a Jeffreys-type Gaussian discrepancy on position states.
% The output alpha is used as a reliability/corroboration factor that
% modulates measurement trust.
%
% This is an ALARM-inspired engineering mechanism operating at the EKF
% track level. It should not be confused with exact labeled-RFS fusion or
% exact multi-object Bayesian consensus.

alpha_trk = compute_alpha_tracks_jeffreys(tt_predict, tt, filter);

% =========================================================================
% gating + loglik using R_eff per track
% =========================================================================
gated = cell(nT,1);
loglik = -inf(nT,m);
for i=1:nT
    [gated{i}, loglik(i,:)] = gate_and_loglik_ekf_alarm(model, filter, tt_predict{i}, Z, alpha_trk(i));
end

% =========================================================================
% SINGLE GLOBAL HUNGARIAN WITH GENTLE ANTI-SWITCH BIAS
% =========================================================================
assign = global_hungarian_assign_continuity(loglik, gated, tt_predict, Z, filter);

% --- update tracks
for i=1:nT
    j = assign(i);
    if j > 0
        a = alpha_trk(i);
        [m_upd, P_upd] = ekf_update_polar_alarm(model, filter, tt_predict{i}.x, tt_predict{i}.w, Z(:,j), a);
        tt_predict{i}.x = m_upd;
        tt_predict{i}.w = P_upd;

        rnew = score_hit(tt_predict{i}.r);
        if filter.alarm_hit_penalty
            rnew = rnew * (0.5 + 0.5*a);
        end
        tt_predict{i}.r = rnew;

        hit0 = getfield_default(tt_predict{i},'hit_streak',0);
        mat0 = getfield_default(tt_predict{i},'maturity',0);

        tt_predict{i}.hit_streak = hit0 + 1;
        tt_predict{i}.miss_streak = 0;
        tt_predict{i}.maturity = mat0 + filter.maturity_hit_inc;

    else
        rnew = score_missed(tt_predict{i}.r, model.P_D);
        if filter.alarm_miss_penalty
            a = alpha_trk(i);
            rnew = rnew * (0.85 + 0.15*a);
        end
        tt_predict{i}.r = rnew;

        miss0 = getfield_default(tt_predict{i},'miss_streak',0);
        hit0  = getfield_default(tt_predict{i},'hit_streak',0);
        mat0  = getfield_default(tt_predict{i},'maturity',0);

        tt_predict{i}.miss_streak = miss0 + 1;
        tt_predict{i}.hit_streak  = max(hit0 - 1, 0);
        tt_predict{i}.maturity    = max(mat0 - filter.maturity_miss_decay, 0);
    end
end

% -------------------------------------------------------------------------
% HARD PRUNE AGAIN AFTER UPDATE TO PREVENT ZOMBIE REAPPEARANCE
% -------------------------------------------------------------------------
tt_out = prune_dead_tracks_hard(tt_predict, filter);
diaginfo.T_posterior = length(tt_out);
end


% =========================================================================
% HARD STALE-TRACK PRUNING
% =========================================================================
function tt_out = prune_dead_tracks_hard(tt_in, filter)
if ~filter.prune_use || isempty(tt_in)
    tt_out = tt_in;
    return;
end

keep = true(numel(tt_in),1);

for i = 1:numel(tt_in)
    trk = tt_in{i};

    r_i    = getfield_default(trk,'r',0);
    age_i  = getfield_default(trk,'age',0);
    hit_i  = getfield_default(trk,'hit_streak',0);
    miss_i = getfield_default(trk,'miss_streak',0);
    mat_i  = getfield_default(trk,'maturity',0);

    is_birthlike = (hit_i <= filter.prune_birth_hit_max) && ...
                   (mat_i <= filter.prune_birth_maturity_max);

    is_mature = (mat_i >= filter.protect_min_maturity) || ...
                (hit_i >= filter.protect_min_hits && age_i >= filter.protect_min_age);

    kill = false;

    % hard low-score kill
    if r_i < filter.prune_r_hard
        kill = true;
    end

    % birth-like tracks should die very quickly if they are not catching on
    if ~kill && is_birthlike && miss_i > filter.prune_birth_miss_max
        kill = true;
    end

    % weak tracks should not linger
    if ~kill && ~is_mature && miss_i > filter.prune_weak_miss_max
        kill = true;
    end

    % mature tracks get more tolerance, but still not infinite
    if ~kill && is_mature && miss_i > filter.prune_mature_miss_max
        kill = true;
    end

    % old but still weak = stale / zombie-like
    if ~kill && ~is_mature && age_i > filter.prune_very_old_weak_age && r_i < filter.prune_very_old_weak_r
        kill = true;
    end

    keep(i) = ~kill;
end

tt_out = tt_in(keep);
end


% =========================================================================
% PRIOR BIRTH GENERATION
% =========================================================================
function tt_birth = make_prior_births(model, filter, k)

nb = length(model.r_birth);
tt_birth = cell(nb,1);

for b = 1:nb
    tt_birth{b}.r = filter.birth_r0;
    tt_birth{b}.x = model.m_birth{b}(:,1);
    tt_birth{b}.w = model.P_birth{b}(:,:,1);
    tt_birth{b}.l = [k; filter.label_base + b]; %#ok<NASGU>
    tt_birth{b}.age = filter.birth_age0;
    tt_birth{b}.hit_streak = filter.birth_hit0;
    tt_birth{b}.miss_streak = filter.birth_miss0;
    tt_birth{b}.maturity = filter.birth_maturity0;
    tt_birth{b}.prev_pos = tt_birth{b}.x([1;3]);
    tt_birth{b}.prev_vel = tt_birth{b}.x([2;4]);
end
end


% =========================================================================
% EKF-ALARM: track reliability alpha via Jeffreys assignment (position only)
% =========================================================================
function alpha = compute_alpha_tracks_jeffreys(tt_predict, tt, filter)
nT = numel(tt_predict);
alpha = ones(nT,1);

num_tables = numel(tt);
if num_tables==0
    return;
end

effective_neighbors = 0;
vote_counts = zeros(nT,1);

[Mi, Pi] = pack_pos_gaussians(tt_predict, filter);

for idx = 1:num_tables
    if ~isfield(tt(idx),'tt_lmb_update') || isempty(tt(idx).tt_lmb_update)
        if filter.alarm_count_only_nonempty
            continue;
        else
            effective_neighbors = effective_neighbors + 1;
            continue;
        end
    end

    nei_tracks = tt(idx).tt_lmb_update;
    if isempty(nei_tracks)
        if filter.alarm_count_only_nonempty
            continue;
        else
            effective_neighbors = effective_neighbors + 1;
            continue;
        end
    end

    effective_neighbors = effective_neighbors + 1;

    [Mj, Pj] = pack_pos_gaussians(nei_tracks, filter);
    if isempty(Mj)
        continue;
    end

    C = jeffreys_cost_matrix(Mi, Pi, Mj, Pj);
    pi = hungarian_with_null(C, filter.alarm_c_null, filter.alarm_cost_gate);

    vote_counts = vote_counts + (pi(:) > 0);
end

if effective_neighbors > 0
    alpha = vote_counts / effective_neighbors;
else
    alpha(:) = 1;
end
end

function [M, P] = pack_pos_gaussians(tracks, filter)
n = numel(tracks);
M = zeros(2,n);
P = zeros(2,2,n);
keep = true(1,n);

for i=1:n
    trk = tracks{i};
    if ~isfield(trk,'x') || isempty(trk.x) || ~isfield(trk,'w') || isempty(trk.w)
        keep(i) = false;
        continue;
    end
    M(:,i) = trk.x([1 3]);
    Pi = trk.w([1 3],[1 3]);
    Pi = 0.5*(Pi + Pi.');
    Pi = Pi + filter.alarm_pos_epsP * eye(2);
    P(:,:,i) = Pi;
end

M = M(:,keep);
P = P(:,:,keep);
end

function C = jeffreys_cost_matrix(Mi, Pi, Mj, Pj)
ni = size(Mi,2);
nj = size(Mj,2);
C = zeros(ni,nj);

for a=1:ni
    m1 = Mi(:,a); P1 = Pi(:,:,a);
    iP1 = inv(P1); %#ok<MINV>
    for b=1:nj
        m2 = Mj(:,b); P2 = Pj(:,:,b);
        iP2 = inv(P2); %#ok<MINV>
        d = m1 - m2;
        C(a,b) = 0.5*( trace(iP2*P1) + trace(iP1*P2) - 4 + (d.'*(iP1+iP2)*d) );
    end
end
end

function pi = hungarian_with_null(C, c_null, cost_gate)
ni = size(C,1);
nj = size(C,2);

C2 = C;
C2(C2 > cost_gate) = c_null;

nCols = nj + ni;
n = max(ni, nCols);

A = c_null * ones(n,n);
A(1:ni,1:nj) = C2;

for a=1:ni
    A(a, nj+a) = c_null;
end

for r=(ni+1):n
    A(r,:) = 0;
end

assign = hungarian_square(A);

pi = zeros(ni,1);
for a=1:ni
    b = assign(a);
    if b>=1 && b<=nj && A(a,b) < c_null
        pi(a) = b;
    else
        pi(a) = 0;
    end
end
end


% =========================================================================
% EKF CT prediction
% =========================================================================
function [m_pred, P_pred] = ekf_predict_ct(model, m, P, Qk)
f = @(x) ct_step(x, model.T);
m_pred = f(m);
F = numjac(f, m);
P_pred = F*P*F.' + Qk;
P_pred = 0.5*(P_pred + P_pred.');
end

function x_next = ct_step(x, T)
px = x(1); vx = x(2);
py = x(3); vy = x(4);
w  = x(5);
wt = w*T;

if abs(w) < 1e-6
    pxn = px + vx*T;
    pyn = py + vy*T;
    vxn = vx;
    vyn = vy;
else
    sw = sin(wt); cw = cos(wt);
    pxn = px + (sw/w)*vx - ((1 - cw)/w)*vy;
    pyn = py + ((1 - cw)/w)*vx + (sw/w)*vy;
    vxn =  cw*vx - sw*vy;
    vyn =  sw*vx + cw*vy;
end

x_next = [pxn; vxn; pyn; vyn; w];
end


% =========================================================================
% EKF update with ALARM-modulated measurement covariance
% =========================================================================
function [m_upd, P_upd] = ekf_update_polar_alarm(model, filter, m_pred, P_pred, z, alpha)
h = @(x) h_obs_like_gen_observation_fn(x);
z_pred = h(m_pred);
H = numjac(h, m_pred);

a = max(alpha, filter.alarm_eps_alpha);
R_eff = model.R / a;

S = H*P_pred*H.' + R_eff;
S = 0.5*(S + S.');
nu = z - z_pred;
nu(1) = wrapToPi_safe(nu(1));
K = (P_pred*H.') / S;

m_upd = m_pred + K*nu;
P_upd = (eye(size(P_pred)) - K*H) * P_pred;
P_upd = 0.5*(P_upd + P_upd.');
end

function [gate_idx, loglik_row] = gate_and_loglik_ekf_alarm(model, filter, trk, Z, alpha)
m = size(Z,2);
loglik_row = -inf(1,m);
gate_idx = [];

h = @(x) h_obs_like_gen_observation_fn(x);
z_pred = h(trk.x);
H = numjac(h, trk.x);

a = max(alpha, filter.alarm_eps_alpha);
R_eff = model.R / a;

S = H*trk.w*H.' + R_eff;
S = 0.5*(S + S.');
Sinv = inv(S); %#ok<MINV>

for j=1:m
    nu = Z(:,j) - z_pred;
    nu(1) = wrapToPi_safe(nu(1));
    d2 = nu.' * Sinv * nu;

    if ~filter.gate_flag || d2 <= filter.gate_chi2
        gate_idx(end+1) = j; %#ok<AGROW>
        lj = gauss_pdf(nu, zeros(model.z_dim,1), S);
        lj = max(lj, filter.max_innov_lik_floor);
        loglik_row(j) = log(lj);
    end
end
end

function z = h_obs_like_gen_observation_fn(x)
P = x([1 3]);
bearing = atan2(P(1), P(2));
range   = sqrt(sum(P.^2));
z = [bearing; range];
end

function Pz = polar_to_cart_like_your_model(Z)
theta = Z(1,:);
r = Z(2,:);
Pz = [r.*sin(theta); r.*cos(theta)];
end


% =========================================================================
% Assignment (Hungarian + continuity + age + maturity + gentle soft bias)
% =========================================================================
function assign = global_hungarian_assign_continuity(loglik, gated, tracks, Z, filter)
[nT,m] = size(loglik);
assign = zeros(nT,1);

if nT==0 || m==0
    return;
end

Pz = polar_to_cart_like_your_model(Z);

nRealCols  = m;
nDummyCols = nT;
nCols      = nRealCols + nDummyCols;
nRows      = nT;
n          = max(nRows, nCols);

C = filter.assign_infeasible_cost * ones(n,n);

for i=1:nT
    trk = tracks{i};

    age_i = getfield_default(trk,'age',1);
    hits_i = getfield_default(trk,'hit_streak',0);
    mat_i  = getfield_default(trk,'maturity',0);

    ageBonus = 0;
    if filter.age_use
        ageBonus = filter.age_bonus_lambda * min(age_i / max(filter.age_confirm_hits,1), 1);
        ageBonus = min(ageBonus, filter.age_bonus_cap);
    end

    maturityBonus = 0;
    if filter.maturity_use
        maturityBonus = filter.maturity_bonus_lambda * min(mat_i / max(filter.maturity_confirm,1e-9), 1);
        maturityBonus = min(maturityBonus, filter.maturity_bonus_cap);
    end

    protectBonus = 0;
    if filter.protect_confirmed
        if (hits_i >= filter.protect_min_hits && age_i >= filter.protect_min_age) || ...
           (mat_i >= filter.protect_min_maturity)
            protectBonus = filter.protect_bonus_extra;
        end
    end

    weakPenalty = 0;
    if mat_i < filter.maturity_confirm
        weakPenalty = filter.weak_track_cost;
    end

    if isempty(gated{i}), continue; end
    for jj=1:numel(gated{i})
        j = gated{i}(jj);
        s = loglik(i,j);
        if ~isfinite(s)
            continue;
        end

        baseCost = -s;

        [contPenalty, dirPenalty] = compute_motion_penalties(trk, Pz(:,j), filter);

        softReservePenalty = 0;
        if filter.soft_reserve_use && (mat_i < filter.maturity_confirm)
            if is_near_recent_mature_track(Pz(:,j), tracks, i, filter)
                softReservePenalty = filter.soft_reserve_penalty;
            end
        end

        totalCost = baseCost + ...
                    filter.cont_lambda * contPenalty + ...
                    filter.dir_lambda  * dirPenalty + ...
                    weakPenalty + softReservePenalty - ...
                    ageBonus - maturityBonus - protectBonus;

        C(i,j) = totalCost;
    end
end

missCost = -filter.assign_unmatched_loglik;
for i=1:nT
    trk = tracks{i};
    age_i = getfield_default(trk,'age',1);
    hits_i = getfield_default(trk,'hit_streak',0);
    mat_i  = getfield_default(trk,'maturity',0);

    ageBonusMiss = 0;
    if filter.age_use
        ageBonusMiss = 0.25 * min(age_i / max(filter.age_confirm_hits,1), 1);
    end

    maturityBonusMiss = 0;
    if filter.maturity_use
        maturityBonusMiss = 0.35 * min(mat_i / max(filter.maturity_confirm,1e-9), 1);
    end

    protectBonusMiss = 0;
    if filter.protect_confirmed
        if (hits_i >= filter.protect_min_hits && age_i >= filter.protect_min_age) || ...
           (mat_i >= filter.protect_min_maturity)
            protectBonusMiss = 0.25;
        end
    end

    C(i, m+i) = max(0, missCost - ageBonusMiss - maturityBonusMiss - protectBonusMiss);
end

for r=(nRows+1):n
    C(r,:) = 0;
end
for c=(nCols+1):n
    C(:,c) = min(C(:,c), 0);
end

sol = hungarian_square(C);

for i=1:nT
    j = sol(i);
    if j>=1 && j<=m && isfinite(loglik(i,j))
        assign(i) = j;
    else
        assign(i) = 0;
    end
end
end


% =========================================================================
% Soft reservation helper
% =========================================================================
function tf = is_near_recent_mature_track(zcart, tracks, self_idx, filter)
tf = false;

for t = 1:numel(tracks)
    if t == self_idx
        continue;
    end

    trk = tracks{t};
    if ~is_recent_mature_track(trk, filter)
        continue;
    end

    if ~isfield(trk,'x') || isempty(trk.x)
        continue;
    end

    p = trk.x([1;3]);
    if norm(zcart - p) <= filter.soft_reserve_R
        tf = true;
        return;
    end
end
end

function tf = is_recent_mature_track(trk, filter)
hits_i = getfield_default(trk,'hit_streak',0);
age_i  = getfield_default(trk,'age',0);
mat_i  = getfield_default(trk,'maturity',0);
miss_i = getfield_default(trk,'miss_streak',0);

is_mature = ((hits_i >= filter.protect_min_hits && age_i >= filter.protect_min_age) || ...
             (mat_i >= filter.protect_min_maturity));

tf = is_mature && (miss_i <= filter.soft_reserve_max_miss);
end


% =========================================================================
% Motion penalties: position continuity + very mild direction consistency
% =========================================================================
function [contPenalty, dirPenalty] = compute_motion_penalties(trk, zcart, filter)

cur_pos = trk.x([1;3]);
cur_vel = trk.x([2;4]);

if isfield(trk,'prev_pos') && ~isempty(trk.prev_pos)
    prev_pos = trk.prev_pos;
else
    prev_pos = cur_pos;
end

if isfield(trk,'prev_vel') && ~isempty(trk.prev_vel)
    prev_vel = trk.prev_vel;
else
    prev_vel = cur_vel;
end

ref1 = cur_pos;
ref2 = cur_pos + 0.5*(cur_pos - prev_pos);

if norm(cur_vel) > 1e-6
    ref3 = cur_pos + 0.25*cur_vel;
else
    ref3 = cur_pos;
end

if norm(prev_vel) > 1e-6
    ref4 = cur_pos + 0.10*prev_vel;
else
    ref4 = cur_pos;
end

d1 = norm(zcart - ref1);
d2 = norm(zcart - ref2);
d3 = norm(zcart - ref3);
d4 = norm(zcart - ref4);

d = min([d1, d2, d3, d4]);

contPenalty = (d / filter.cont_sigma_pos)^2;
contPenalty = min(contPenalty, filter.cont_max_penalty);

dirPenalty = 0;
if ~filter.dir_use
    return;
end

vref = cur_vel;
if norm(vref) < 1e-6 && norm(prev_vel) > 1e-6
    vref = prev_vel;
end

dispvec = zcart - cur_pos;
spd = norm(vref);
dd  = norm(dispvec);

if spd < filter.dir_speed_floor || dd < 1e-6
    dirPenalty = filter.dir_penalty_stop;
    return;
end

cang = (vref(:)' * dispvec(:)) / (spd * dd);
cang = max(-1,min(1,cang));

if cang < filter.dir_cos_bad
    dirPenalty = filter.dir_penalty_bad;
elseif cang < 0.4
    dirPenalty = filter.dir_penalty_side;
else
    dirPenalty = 0;
end
end


% =========================================================================
% Hungarian algorithm for square cost matrix
% =========================================================================
function assign = hungarian_square(costMat)
n = size(costMat,1);
C = costMat;

for i=1:n
    C(i,:) = C(i,:) - min(C(i,:));
end

for j=1:n
    C(:,j) = C(:,j) - min(C(:,j));
end

mask = zeros(n,n);   % 0 none, 1 star, 2 prime
rowCover = false(n,1);
colCover = false(n,1);

for i=1:n
    for j=1:n
        if abs(C(i,j)) < 1e-12 && ~rowCover(i) && ~colCover(j)
            mask(i,j)=1;
            rowCover(i)=true;
            colCover(j)=true;
        end
    end
end
rowCover(:)=false;
colCover(:)=false;

step = 4;
Z0_r = 0; Z0_c = 0;
path = zeros(n*2,2);

while true
    switch step
        case 4
            for j=1:n
                if any(mask(:,j)==1)
                    colCover(j)=true;
                end
            end
            if sum(colCover)==n
                step = 7;
            else
                step = 5;
            end

        case 5
            [r,c] = findZero(C,rowCover,colCover);
            if r==0
                step = 6;
            else
                mask(r,c)=2;
                starCol = find(mask(r,:)==1,1);
                if ~isempty(starCol)
                    rowCover(r)=true;
                    colCover(starCol)=false;
                else
                    step = 55;
                    Z0_r = r; Z0_c = c;
                end
            end

        case 55
            count=1;
            path(count,:) = [Z0_r Z0_c];
            done=false;
            while ~done
                r = find(mask(:,path(count,2))==1,1);
                if isempty(r)
                    done=true;
                else
                    count=count+1;
                    path(count,:) = [r path(count-1,2)];
                    c = find(mask(path(count,1),:)==2,1);
                    count=count+1;
                    path(count,:) = [path(count-1,1) c];
                end
            end

            for k2=1:count
                if mask(path(k2,1),path(k2,2))==1
                    mask(path(k2,1),path(k2,2))=0;
                else
                    mask(path(k2,1),path(k2,2))=1;
                end
            end

            rowCover(:)=false;
            colCover(:)=false;
            mask(mask==2)=0;
            step = 4;

        case 6
            minval = minUncovered(C,rowCover,colCover);
            for i=1:n
                if rowCover(i)
                    C(i,:) = C(i,:) + minval;
                end
            end
            for j=1:n
                if ~colCover(j)
                    C(:,j) = C(:,j) - minval;
                end
            end
            step = 5;

        case 7
            assign = zeros(n,1);
            for i=1:n
                j = find(mask(i,:)==1,1);
                if ~isempty(j)
                    assign(i)=j;
                else
                    assign(i)=1;
                end
            end
            return;
    end
end
end

function [r,c] = findZero(C,rowCover,colCover)
n = size(C,1);
r=0; c=0;
tol = 1e-12;
for i=1:n
    if rowCover(i), continue; end
    for j=1:n
        if colCover(j), continue; end
        if abs(C(i,j)) < tol
            r=i; c=j;
            return;
        end
    end
end
end

function m = minUncovered(C,rowCover,colCover)
n = size(C,1);
m = inf;
for i=1:n
    if rowCover(i), continue; end
    for j=1:n
        if colCover(j), continue; end
        if C(i,j) < m
            m = C(i,j);
        end
    end
end
if ~isfinite(m), m = 0; end
end


% =========================================================================
% Track confidence logic
% =========================================================================
% -------------------------------------------------------------------------
% TRACK CONFIDENCE SCORE UPDATE
% -------------------------------------------------------------------------
% The variable r is used here as a heuristic confidence score for track
% management. It is NOT a strict Bernoulli existence probability in the
% formal LMB/GLMB sense, even though legacy notation may suggest that.


function r = score_hit(r)
r = min(0.999999, r + 0.20*(1-r));
end

function r = score_missed(r, PD)
r = max(0, r*(1 - 0.10*PD));
end


% =========================================================================
% cleanup / cap
% =========================================================================
% -------------------------------------------------------------------------
% LEGACY CLEANUP FUNCTION NAME
% -------------------------------------------------------------------------
% The name "clean_lmb" is preserved only for pipeline compatibility.
% Functionally, this routine performs score-based EKF track pruning and
% capping; it is not an LMB-specific truncation step in the strict RFS
% sense.

function tt_out = clean_lmb(tt_in, filter)
if isempty(tt_in), tt_out = tt_in; return; end

tt_in = prune_dead_tracks_hard(tt_in, filter);

rv = get_rvals(tt_in);
idxkeep = find(rv > filter.track_threshold);
tt_out = tt_in(idxkeep);

if length(tt_out) > filter.T_max
    rv2 = get_rvals(tt_out);
    [~,idx] = sort(rv2,'descend');
    tt_out = tt_out(idx(1:filter.T_max));
end
end

function rv = get_rvals(tt)
rv = zeros(numel(tt),1);
for i=1:numel(tt)
    rv(i) = tt{i}.r;
end
end


% =========================================================================
% estimation
% =========================================================================
% -------------------------------------------------------------------------
% ESTIMATE EXTRACTION
% -------------------------------------------------------------------------
% This function extracts the reported target states from the maintained EKF
% track list using score and maturity thresholds.
%
% The output label field L is preserved for compatibility with downstream
% plotting/evaluation code, but these labels should be interpreted as
% legacy metadata rather than strict labeled-RFS identities.

function [X,N,L] = extract_estimates_score(tt_lmb, model, filter)
if isempty(tt_lmb)
    X = zeros(model.x_dim,0);
    N = 0;
    L = zeros(2,0);
    return;
end

rv = get_rvals(tt_lmb);
mv = zeros(numel(tt_lmb),1);
for i=1:numel(tt_lmb)
    mv(i) = getfield_default(tt_lmb{i},'maturity',0);
end

idx = find((rv >= filter.est_r_min) & (mv >= filter.est_min_maturity));
if isempty(idx)
    X = zeros(model.x_dim,0);
    N = 0;
    L = zeros(2,0);
    return;
end

[~,ord] = sort(rv(idx),'descend');
idx = idx(ord);

N = min(numel(idx), filter.est_maxN);
X = zeros(model.x_dim, N);
L = zeros(2, N);

for n=1:N
    X(:,n) = tt_lmb{idx(n)}.x;
    L(:,n) = tt_lmb{idx(n)}.l;
end
end


% =========================================================================
% diagnostics
% =========================================================================
function display_diaginfo(tt_lmb,k,est,filter,T_predict,T_posterior,T_clean)
if isempty(tt_lmb)
    score_mean = 0;
    score_max  = 0;
    mean_age = 0;
    mean_hits = 0;
    mean_maturity = 0;
else
    rv = get_rvals(tt_lmb);
    score_mean = mean(rv);
    score_max  = max(rv);

    ages = zeros(numel(tt_lmb),1);
    hits = zeros(numel(tt_lmb),1);
    mats = zeros(numel(tt_lmb),1);

    for i=1:numel(tt_lmb)
        ages(i) = getfield_default(tt_lmb{i},'age',1);
        hits(i) = getfield_default(tt_lmb{i},'hit_streak',0);
        mats(i) = getfield_default(tt_lmb{i},'maturity',0);
    end

    mean_age = mean(ages);
    mean_hits = mean(hits);
    mean_maturity = mean(mats);
end

if ~strcmp(filter.run_flag,'silence')
    disp([' time= ',num2str(k),...
        ' score_mean=' num2str(score_mean,3),...
        ' score_max='  num2str(score_max,3),...
        ' age_mean='   num2str(mean_age,3),...
        ' hit_mean='   num2str(mean_hits,3),...
        ' mat_mean='   num2str(mean_maturity,3),...
        ' #est=' num2str(est.N(k),4),...
        ' #trax pred=' num2str(T_predict,4),...
        ' #trax post=' num2str(T_posterior,4),...
        ' #trax kept=' num2str(T_clean,4)]);
end
end


% =========================================================================
% numerical jacobian
% =========================================================================
function J = numjac(fun, x0)
y0 = fun(x0);
ny = numel(y0);
nx = numel(x0);
J = zeros(ny,nx);
eps0 = 1e-5;

for i=1:nx
    dx = zeros(nx,1);
    step = eps0*(1+abs(x0(i)));
    dx(i)=step;
    yp = fun(x0+dx);
    ym = fun(x0-dx);
    J(:,i) = (yp-ym)/(2*step);
end
end


% =========================================================================
% gaussian pdf utility + helpers
% =========================================================================
function val = gauss_pdf(x, m, S)
d = numel(x);
xc = x - m;
[U,p] = chol(S);
if p~=0
    S = S + 1e-9*eye(d);
    U = chol(S);
end
q = U'\xc;
quad = q.'*q;
logdet = 2*sum(log(diag(U)));
val = exp(-0.5*(d*log(2*pi) + logdet + quad));
end

function a = wrapToPi_safe(a)
a = mod(a + pi, 2*pi) - pi;
end

function x = chi2inv_safe(p, dof)
if exist('chi2inv','file') == 2
    x = chi2inv(p, dof);
    return;
end
if dof == 2
    x = -2*log(max(1-p, 1e-12));
else
    x = 9.21;
end
end

function v = getfield_default(s, f, d)
if isfield(s,f)
    v = s.(f);
else
    v = d;
end
end

% -------------------------------------------------------------------------
% REPOSITORY NOTE
% -------------------------------------------------------------------------
% If this file is distributed publicly, it should be described as an
% EKF-based multi-target tracker implemented within a legacy LMB/GLMB-style
% interface for compatibility with an existing codebase.