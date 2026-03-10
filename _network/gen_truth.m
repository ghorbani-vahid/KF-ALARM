function truth= gen_truth(model,iter_num, attack)

rng(42);
% Variables
truth.K= iter_num;                   % length of data/number of scans
truth.X= cell(truth.K,1);             % ground truth for states of targets
truth.N= zeros(truth.K,1);            % ground truth for number of targets
truth.L= cell(truth.K,1);             % ground truth for labels of targets (k,i)
truth.track_list= cell(truth.K,1);    % absolute index target identities (plotting)
truth.total_tracks= 0;                % total number of appearing tracks

nbirths= 4;
wturn = 2*pi/180;

if attack.scenario== "none"
    
xstart(:,1)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(2)  = 5;    tdeath(2)  = truth.K+1-5;
xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 7;    tdeath(3)  = truth.K+1-3;
xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 9;    tdeath(4)  = 66-1;

nbirths= 4;
end

if attack.scenario== "none-for-replay"
    
xstart(:,1)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(2)  = 5;    tdeath(2)  = truth.K+1-5;
xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 7;    tdeath(3)  = truth.K+1-3;
xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 9;    tdeath(4)  = 66-1;

nbirths= 4;
end

if attack.scenario== "replay" 

xstart(:,1)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(2)  = 5;    tdeath(2)  = truth.K+1-5;
xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 7;    tdeath(3)  = truth.K+1-3;
xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 9;    tdeath(4)  = 66-1;

%ghosts
xstart(:,5)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(5)  = 1+10;     tdeath(5)  = truth.K+1;  % Replay after 10 time steps
xstart(:,6)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(6)  = 5+10;    tdeath(6)  = truth.K+1-5;
xstart(:,7)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(7)  = 7+10;    tdeath(7)  = truth.K+1-3;
xstart(:,8)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(8)  = 9+10;    tdeath(8)  = 66-1;

nbirths= 8;
end

if attack.scenario== "flooding" 

xstart(:,1)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(2)  = 5;    tdeath(2)  = truth.K+1-5;
xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 7;    tdeath(3)  = truth.K+1-3;
xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 9;    tdeath(4)  = 66-1;


%ghosts
xstart(:,5) = [ 1000+3.9676; -10; 1500-11.6457; -10; wturn/8 ];        tbirth(5)  = 10;     tdeath(5)  = truth.K+1;
xstart(:,6) = [ -250-5.9857;  20; 1000+11.5102; 3; -wturn/3 ];         tbirth(6)  = 10;    tdeath(6)  = truth.K+1;
xstart(:,7) = [ -1500-7.2806; 11; 250+6.8993; 10; -wturn/2 ];          tbirth(7)  = 10;    tdeath(7)  = truth.K+1;
xstart(:,8) = [ -1500+1; 43; 250+1; 0; 0 ];                            tbirth(8)  = 10;    tdeath(8)  = 66;
xstart(:,9)  = [ 250-3.8676; 11; 750-11.0747; 5; wturn/4 ];             tbirth(9)  = 20;    tdeath(9)  = 80;
xstart(:,10) = [ 250-3.7676; 11; 750-11.1747; 5; wturn/4 ];             tbirth(10)  = 20;    tdeath(10)  = 80;
xstart(:,11)  = [ -250+7.3806; -12; 1000-6.7993; -12; wturn/2 ];         tbirth(11)  = 40;    tdeath(11)  = truth.K+1;
xstart(:,12)  = [ 1000; 0; 1500; -10; wturn/4 ];                         tbirth(12)  = 40;    tdeath(12)  = truth.K+1;
xstart(:,13)  = [ 250; -50; 750; 0; -wturn/4 ];                          tbirth(13)  = 40;    tdeath(13)  = 80;

xstart(:,14) = [ -250+7.4806; -12; 1000-6.6993; -12; wturn/2 ];         tbirth(14)  = 40;    tdeath(14)  = truth.K+1;
xstart(:,15) = [ 1000+1; 0; 1500+1; -10; wturn/4 ];                     tbirth(15)  = 40;    tdeath(15)  = truth.K+1;
xstart(:,16) = [ 250; -50; 750+2; 0; -wturn/4 ];                      tbirth(16)  = 40;    tdeath(16)  = 80;
xstart(:,17)  = [ 1000; -50; 1500; 0; -wturn/4 ];                        tbirth(17)  = 60;    tdeath(17)  = truth.K+1;
xstart(:,18) = [ 250; -40; 750; 25; wturn/4 ];                          tbirth(18) = 60;    tdeath(18) = truth.K+1;
xstart(:,19) = [ 1000-1; -50; 1500-2; 0; -wturn/4 ];                    tbirth(19)  = 60;    tdeath(19)  = truth.K+1;
xstart(:,20) = [ 250+2; -40; 750+3; 25; wturn/4 ];                      tbirth(20)  = 60;    tdeath(20)  = truth.K+1;

xstart(:,21)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(21)  = 1+10;     tdeath(21)  = truth.K+1;
xstart(:,22)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(22)  = 5+10;    tdeath(22)  = truth.K+1;
xstart(:,23)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(23)  = 7+10;    tdeath(23)  = truth.K+1;
xstart(:,24)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(24)  = 9+10;    tdeath(24)  = 66-1+10;


nbirths= 24;
end



if attack.scenario=="deception" && attack.ghost_num==4

xstart(:,1)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(2)  = 5;    tdeath(2)  = truth.K+1-5;
xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 7;    tdeath(3)  = truth.K+1-3;
xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 9;    tdeath(4)  = 66-1;


%ghosts
xstart(:,5) = [ 1000+3.9676; -10; 1500-11.6457; -10; wturn/8 ];        tbirth(5)  = 10;     tdeath(5)  = truth.K+1;
xstart(:,6) = [ -250-5.9857;  20; 1000+11.5102; 3; -wturn/3 ];         tbirth(6)  = 10;    tdeath(6)  = truth.K+1;
xstart(:,7) = [ -1500-7.2806; 11; 250+6.8993; 10; -wturn/2 ];          tbirth(7)  = 10;    tdeath(7)  = truth.K+1;
xstart(:,8) = [ -1500+1; 43; 250+1; 0; 0 ];                            tbirth(8)  = 10;    tdeath(8)  = 66;

nbirths= 8;
end

if attack.scenario=="deception" && attack.ghost_num==6

xstart(:,1)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(2)  = 5;    tdeath(2)  = truth.K+1-5;
xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 7;    tdeath(3)  = truth.K+1-3;
xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 9;    tdeath(4)  = 66-1;



%ghosts
xstart(:,5) = [ 1000+3.9676; -10; 1500-11.6457; -10; wturn/8 ];        tbirth(5)  = 10;     tdeath(5)  = truth.K+1;
xstart(:,6) = [ -250-5.9857;  20; 1000+11.5102; 3; -wturn/3 ];         tbirth(6)  = 10;    tdeath(6)  = truth.K+1;
xstart(:,7) = [ -1500-7.2806; 11; 250+6.8993; 10; -wturn/2 ];          tbirth(7)  = 10;    tdeath(7)  = truth.K+1;
xstart(:,8) = [ -1500+1; 43; 250+1; 0; 0 ];                            tbirth(8)  = 10;    tdeath(8)  = 66;
xstart(:,9)  = [ 250-3.8676; 11; 750-11.0747; 5; wturn/4 ];             tbirth(9)  = 20;    tdeath(9)  = 80;
xstart(:,10) = [ 250-3.7676; 11; 750-11.1747; 5; wturn/4 ];             tbirth(10)  = 20;    tdeath(10)  = 80;

nbirths= 10;
end


if attack.scenario=="deception" && attack.ghost_num==12

xstart(:,1)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(2)  = 5;    tdeath(2)  = truth.K+1-5;
xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 7;    tdeath(3)  = truth.K+1-3;
xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 9;    tdeath(4)  = 66-1;



%ghosts
xstart(:,5) = [ 1000+3.9676; -10; 1500-11.6457; -10; wturn/8 ];        tbirth(5)  = 10;     tdeath(5)  = truth.K+1;
xstart(:,6) = [ -250-5.9857;  20; 1000+11.5102; 3; -wturn/3 ];         tbirth(6)  = 10;    tdeath(6)  = truth.K+1;
xstart(:,7) = [ -1500-7.2806; 11; 250+6.8993; 10; -wturn/2 ];          tbirth(7)  = 10;    tdeath(7)  = truth.K+1;
xstart(:,8) = [ -1500+1; 43; 250+1; 0; 0 ];                            tbirth(8)  = 10;    tdeath(8)  = 66;
xstart(:,9)  = [ 250-3.8676; 11; 750-11.0747; 5; wturn/4 ];             tbirth(9)  = 20;    tdeath(9)  = 80;
xstart(:,10) = [ 250-3.7676; 11; 750-11.1747; 5; wturn/4 ];             tbirth(10)  = 20;    tdeath(10)  = 80;
xstart(:,11)  = [ -250+7.3806; -12; 1000-6.7993; -12; wturn/2 ];         tbirth(11)  = 40;    tdeath(11)  = truth.K+1;
xstart(:,12)  = [ 1000; 0; 1500; -10; wturn/4 ];                         tbirth(12)  = 40;    tdeath(12)  = truth.K+1;
xstart(:,13)  = [ 250; -50; 750; 0; -wturn/4 ];                          tbirth(13)  = 40;    tdeath(13)  = 80;

xstart(:,14) = [ -250+7.4806; -12; 1000-6.6993; -12; wturn/2 ];         tbirth(14)  = 40;    tdeath(14)  = truth.K+1;
xstart(:,15) = [ 1000+1; 0; 1500+1; -10; wturn/4 ];                     tbirth(15)  = 40;    tdeath(15)  = truth.K+1;
xstart(:,16) = [ 250; -50; 750+2; 0; -wturn/4 ];                      tbirth(16)  = 40;    tdeath(16)  = 80;


nbirths= 16;
end
 


if attack.scenario=="deception" && attack.ghost_num==16


xstart(:,1)  = [ 1000+3.8676; -10; 1500-11.7457; -10; wturn/8 ];        tbirth(1)  = 1;     tdeath(1)  = truth.K+1;
xstart(:,2)  = [ -250-5.8857;  20; 1000+11.4102; 3; -wturn/3 ];         tbirth(2)  = 5;    tdeath(2)  = truth.K+1-5;
xstart(:,3)  = [ -1500-7.3806; 11; 250+6.7993; 10; -wturn/2 ];          tbirth(3)  = 7;    tdeath(3)  = truth.K+1-3;
xstart(:,4)  = [ -1500; 43; 250; 0; 0 ];                                tbirth(4)  = 9;    tdeath(4)  = 66-1;


%ghosts
xstart(:,5) = [ 1000+3.9676; -10; 1500-11.6457; -10; wturn/8 ];        tbirth(5)  = 10;     tdeath(5)  = truth.K+1;
xstart(:,6) = [ -250-5.9857;  20; 1000+11.5102; 3; -wturn/3 ];         tbirth(6)  = 10;    tdeath(6)  = truth.K+1;
xstart(:,7) = [ -1500-7.2806; 11; 250+6.8993; 10; -wturn/2 ];          tbirth(7)  = 10;    tdeath(7)  = truth.K+1;
xstart(:,8) = [ -1500+1; 43; 250+1; 0; 0 ];                            tbirth(8)  = 10;    tdeath(8)  = 66;
xstart(:,9)  = [ 250-3.8676; 11; 750-11.0747; 5; wturn/4 ];             tbirth(9)  = 20;    tdeath(9)  = 80;
xstart(:,10) = [ 250-3.7676; 11; 750-11.1747; 5; wturn/4 ];             tbirth(10)  = 20;    tdeath(10)  = 80;
xstart(:,11)  = [ -250+7.3806; -12; 1000-6.7993; -12; wturn/2 ];         tbirth(11)  = 40;    tdeath(11)  = truth.K+1;
xstart(:,12)  = [ 1000; 0; 1500; -10; wturn/4 ];                         tbirth(12)  = 40;    tdeath(12)  = truth.K+1;
xstart(:,13)  = [ 250; -50; 750; 0; -wturn/4 ];                          tbirth(13)  = 40;    tdeath(13)  = 80;

xstart(:,14) = [ -250+7.4806; -12; 1000-6.6993; -12; wturn/2 ];         tbirth(14)  = 40;    tdeath(14)  = truth.K+1;
xstart(:,15) = [ 1000+1; 0; 1500+1; -10; wturn/4 ];                     tbirth(15)  = 40;    tdeath(15)  = truth.K+1;
xstart(:,16) = [ 250; -50; 750+2; 0; -wturn/4 ];                      tbirth(16)  = 40;    tdeath(16)  = 80;
xstart(:,17)  = [ 1000; -50; 1500; 0; -wturn/4 ];                        tbirth(17)  = 60;    tdeath(17)  = truth.K+1;
xstart(:,18) = [ 250; -40; 750; 25; wturn/4 ];                          tbirth(18) = 60;    tdeath(18) = truth.K+1;
xstart(:,19) = [ 1000-1; -50; 1500-2; 0; -wturn/4 ];                    tbirth(19)  = 60;    tdeath(19)  = truth.K+1;
xstart(:,20) = [ 250+2; -40; 750+3; 25; wturn/4 ];                      tbirth(20)  = 60;    tdeath(20)  = truth.K+1;

nbirths= 20;
end
 
 


% Generate the tracks
for targetnum=1:nbirths
    targetstate = xstart(:,targetnum);
    
    for k=tbirth(targetnum):min(tdeath(targetnum),truth.K)
        targetstate = gen_newstate_fn(model,targetstate,'noiseless');
        
        % Compute relative position of the target
        dx = targetstate(1) - model.x_off;
        dy = targetstate(3) - model.y_off;
        range = sqrt(dx^2 + dy^2);
        bearing = atan2(dy, dx);
        
    if range <= model.FoV_r_max 

        truth.X{k} = [truth.X{k}, targetstate];
        truth.track_list{k} = [truth.track_list{k}, targetnum];
        truth.N(k) = truth.N(k) + 1;
   
   end

    end
end

truth.total_tracks =nbirths;