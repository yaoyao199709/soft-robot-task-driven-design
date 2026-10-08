% Define the file path and file name
script_fullpath = mfilename('fullpath');
script_dir = fileparts(script_fullpath);
file_path = fullfile(script_dir, '..', '..', '..', 'data', 'helical', 'optimized_design', 'sm_result.mat');
load(file_path);
csvtable_y = [];
csvtable_x = [];
csvtable_z = [];
for i = 1:size(result_y, 3)

    
    for j = 1:size(result_y, 2)
        if result_y(2, j, i) > 0.5 % remove large theta
            result_y(:, j, i) = [0, 0, 0]';
        end
    end

    data_y = squeeze(result_y(:,:,i))'; % P[kPa] theta[half] M[Nm] 
    csvtable_y = [csvtable_y data_y];
    
    % figure(i)
    % hold on
    % plot(data_y(:,2), data_y(:,3), 'x') % theta vs mom
    % hold off
    
end

for i = 1:size(result_x, 3)
    for j = 1:size(result_x, 2)
        if result_x(2, j, i) > 0.5 % remove large theta
            result_x(:, j, i) = [0, 0, 0]';
        end
    end
    data_x = squeeze(result_x(:,:,i))'; % P[kPa] theta[half] M[Nm] 
    csvtable_x = [csvtable_x data_x];
    % figure(i)
    % hold on
    % plot(data_x(:,2), data_x(:,3), 'x') % theta vs mom
    % hold off

end

for i = 1:size(result_z, 3)
    for j = 1:size(result_z, 2)
        if (result_z(2, j, i) > 0.5) || (result_z(2, j, i) < -0.5) % remove large theta
            result_z(:, j, i) = [0, 0, 0]';
        end
    end
    data_z = squeeze(result_z(:,:,i))'; % P[kPa] theta[half] M[Nm] 
    csvtable_z = [csvtable_z data_z];
    
    % figure(i)
    % hold on
    % plot(data_z(:,2), data_z(:,3), 'x') % theta vs mom
    % hold off

end
%%
output_dir = fullfile(script_dir, 'outputs');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
csvwrite(fullfile(output_dir, 'restructured_sm_res_y.csv'), csvtable_y);
csvwrite(fullfile(output_dir, 'restructured_sm_res_z.csv'), csvtable_z);
csvwrite(fullfile(output_dir, 'restructured_sm_res_x.csv'), csvtable_x);
design_parameters = squeeze(variable_value);
design_parameters = [design_parameters(1,:); design_parameters(3:4,:)];
csvwrite(fullfile(output_dir, 'design_parameters_sm_res.csv'), design_parameters);
