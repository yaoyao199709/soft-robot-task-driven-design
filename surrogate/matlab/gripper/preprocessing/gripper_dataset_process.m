% Define the file path and file name
script_fullpath = mfilename('fullpath');
script_dir = fileparts(script_fullpath);
file_path = fullfile(script_dir, '..', '..', '..', 'data', 'gripper', 'design_sweep', 'gripper_dataset.mat');
load(file_path);

N_sample = size(result_value, 3);

% N_sample = 1;

sample_k_ela_polyfit_result = [];
sample_k_act_polyfit_result = [];
sample_k_ela_polyfit_RMSE_ratio = [];
sample_k_act_polyfit_RMSE_ratio = [];

sample_torque_net_diff = [];
sample_m_act = [];
sample_b_act = [];
sample_m_ela = [];
sample_b_ela = [];

sample_torque_net_theta0_polyfit_result = [];
sample_torque_net_Pmax_theta0 = [];

for n = 1:N_sample
    % first row for free-loading, 
    % the following rows for constrained conditions from 15 to 5 kPa
    row_num = 1 + 15 - 5 + 1; 
    col_num = 15/0.5+1;
    
    P = zeros(row_num,col_num);
    torque_ext = zeros(row_num,col_num);
    U = zeros(row_num,col_num);
    theta = zeros(row_num,col_num);
    % z0 = other(1,2,n);
    r = variable_value(1);
    t = variable_value(2);
    R = variable_value(3);
    z0 = -(2*R-r+t*2);

    % result_name = {'Pressure'; 'External Force'; 'Angular Deflection'; 'x'; 'z'; 'Elastic Strain Energy'; 'Maximum Stress'; 'Maximum Strain'; 'Minimum Element Quality'; 'Other_x0_z0_Ixx_Iyy_Izz_weight'};
    
    row_index = 2;
    col_index = 1;

    for j = 1:size(result_value, 2)
        if result_value(2, j, n) == 0 
            % free-loading data: no external force

            P(1,j) = 1e-3*result_value(1, j, n);
            U(1,j) = result_value(6, j, n);
            theta(1,j) = result_value(3, j, n)*2; % half to one

        else  % constrained data
            
            
            if (col_index == 1) || ((col_index ~= 1) && (result_value(1,j,n) == result_value(1,j-1,n)))
            
                
    
            else % different pressure value
  
                row_index = row_index + 1;
                col_index = 1;
                
                
    
            end

            P(row_index,col_index) = 1e-3*result_value(1, j, n);
            torque_ext(row_index,col_index) = result_value(2, j, n)*1e-3*(result_value(5, j, n)-z0);
            U(row_index,col_index) = result_value(6, j, n);
            theta(row_index,col_index) = result_value(3, j, n)*2; % half to one

            col_index = col_index + 1;
        end
    end

    % get the elastic force, actuation force, external force, and
    % displacement
    k_ela = [];
    theta_ela = [];
    
    % k_act_vs_theta_polyfit_result = [];
    torque_act_vs_theta_polyfit_result = [];
    P_act = [];
    k_act_polyval_RMSE = [];
    k_act_polyval_RMSE_ratio = [];

    for i = 2:size(P, 1) % for each constrained condition with different P
        P_max = P(i,1);
        N_step1 = P_max*2 + 1;
        N_step2 = P_max*2;

    
        dU = U(i, 2:N_step2) - U(i, 1:N_step2-1);
        dtheta = theta(i, 2:N_step2) - theta(i, 1:N_step2-1); 
    
        torque_ela = dU./dtheta; % no counting for the free-loading part
        torque_act = torque_ela + torque_ext(i, 2:N_step2);
        dtorque_ela = torque_ela(2:end) - torque_ela(1:end-1);
        dtorque_act = torque_act(2:end) - torque_act(1:end-1);
        dtorque_ext = torque_ext(i, 2:N_step2) - torque_ext(i, 1:N_step2-1);
    
        % k_ela should be independent of P
        k_ela = [k_ela dtorque_ela./dtheta(2:end)]; % accumlating with P
        theta_ela = [theta_ela theta(i, 3:N_step2)]; % accumlating with P
    
        % k_act depends on P
        % get k_act via finite difference
        k_act = dtorque_act./dtheta(2:end); % for each P
        k_ext = dtorque_ext./dtheta; % one element more than k_ela and k_act, for each P
        
        P_act = [P_act P(i,1)];  %kPa
        % get 1st-order fitted k_act2 wrt theta
        torque_act_vs_theta_polyfit = polyfit(theta(i, 2:N_step2), torque_act, 1);  % for each P
        torque_act_polyval = polyval(torque_act_vs_theta_polyfit, theta(i, 2:N_step2));
        % get the 1st-order fitted k_act2 vs. P
        torque_act_vs_theta_polyfit_result = [torque_act_vs_theta_polyfit_result; torque_act_vs_theta_polyfit]; % accumlating with P
    
    
        k_ext_polyval = polyval(polyfit(theta(i, 2:N_step2), k_ext, 1), theta(i, 2:N_step2)); % for each P
        
    end
    % overall result for each sample

    sample_k_ela_polyfit_result = [sample_k_ela_polyfit_result polyfit(theta_ela, k_ela, 1)]; 
    k_ela_polyval = polyval(polyfit(theta_ela, k_ela, 1), theta_ela);
    
    k_ela_polyval_RMSE = sqrt(mean((k_ela_polyval - k_ela).^2));
    k_ela_polyval_RMSE_ratio = k_ela_polyval_RMSE/max(abs(k_ela))*100;
    sample_k_ela_polyfit_RMSE_ratio = [sample_k_ela_polyfit_RMSE_ratio k_ela_polyval_RMSE_ratio];
    
    sample_k_act_polyfit_result = [sample_k_act_polyfit_result polyfit(P_act, torque_act_vs_theta_polyfit_result(:,1)', 1)]; 
    k_act_polyval = polyval(polyfit(P_act, torque_act_vs_theta_polyfit_result(:,1)', 1), P_act);
    
    k_act_polyval_RMSE = sqrt(mean((k_act_polyval - torque_act_vs_theta_polyfit_result(:,1)').^2));
    k_act_polyval_RMSE_ratio = k_act_polyval_RMSE/max(abs(torque_act_vs_theta_polyfit_result(:,1)'))*100;
    sample_k_act_polyfit_RMSE_ratio = [sample_k_act_polyfit_RMSE_ratio k_act_polyval_RMSE_ratio];

    
    
    % check if the fitted net forces are more or less zero for free-loading
    % the net force for each sample at free-loading pressure can be represented as
    P_free = P(1,:); %kPa
    theta_free = theta(1,:);
    m_act = sample_k_act_polyfit_result(end-1); sample_m_act = [sample_m_act; m_act];
    b_act = sample_k_act_polyfit_result(end); sample_b_act = [sample_b_act; b_act];
    m_ela = sample_k_ela_polyfit_result(end-1); sample_m_ela = [sample_m_ela; m_ela];
    b_ela = sample_k_ela_polyfit_result(end); sample_b_ela = [sample_b_ela; b_ela];
    
    
    % to make the F_net complete, we also need to get F_net_x0
    torque_net_theta0 = [];
    
    for i = 2:size(P, 1)
        theta0 = theta(1,end-(i-2)*2);
        torque_net_theta0 = [torque_net_theta0  0-((m_act*P(i,1) + b_act).*theta0 - (0.5*m_ela*theta0.^2 + b_ela*theta0))];
    end
    
    torque_net_theta0_polyfit_result = polyfit(P(2:end,1), torque_net_theta0, 1);
    sample_torque_net_theta0_polyfit_result = [sample_torque_net_theta0_polyfit_result; torque_net_theta0_polyfit_result(1)];
    
    torque_net = (m_act*P_free + b_act).*theta_free - (0.5*m_ela*theta_free.^2 + b_ela*theta_free) + torque_net_theta0_polyfit_result(1)*P_free;
    
    % in theory, all elements should be zero
    sample_torque_net_diff = [sample_torque_net_diff; torque_net];
    
    sample_torque_net_Pmax_theta0 = [sample_torque_net_Pmax_theta0 torque_net_theta0_polyfit_result(1)*15000];
end

%%
csvtable = [squeeze(variable_value)' sample_m_act sample_b_act sample_m_ela sample_b_ela sample_torque_net_theta0_polyfit_result];
output_dir = fullfile(script_dir, 'outputs');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
csvwrite(fullfile(output_dir, 'poly_surrogate_coefficient.csv'), csvtable);

