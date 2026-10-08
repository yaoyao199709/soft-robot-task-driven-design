% Define the file path and file name
script_fullpath = mfilename('fullpath');
script_dir = fileparts(script_fullpath);
file_path = fullfile(script_dir, '..', '..', '..', 'data', 'helical', 'design_sweep', 'sm_dataset.mat');

load(file_path);
csvtable_y = [];
csvtable_x = [];
csvtable_z = [];
for i = 1:size(result_y, 3)

    
    for j = 1:size(result_y, 2)
        if result_y(2, j, i) > 0.5 % remove large theta
            result_y(:, j, i) = [0, 0, 0]';
        end
        if ((i == 1) || (i == 2) || (i == 7) || (i == 8)) && (result_y(3, j, i) <= -4e-3) && (result_y(2, j, i) > -0.03)
            result_y(:, j, i) = [0, 0, 0]';
        end
    end

    data_y = squeeze(result_y(:,:,i))'; % P[kPa] theta[half] M[Nm] 

    
    figure(i)
    hold on
    plot(data_y(:,2), data_y(:,3), 'x') % theta vs mom
    hold off
    csvtable_y = [csvtable_y data_y];
end

for i = 1:size(result_x, 3)
    for j = 1:size(result_x, 2)
        if result_x(2, j, i) > 0.5 % remove large theta
            result_x(:, j, i) = [0, 0, 0]';
        end
    end
    data_x = squeeze(result_x(:,:,i))'; % P[kPa] theta[half] M[Nm] 
    csvtable_x = [csvtable_x data_x];
end

for i = 1:size(result_z, 3)
    for j = 1:size(result_z, 2)
        if result_z(2, j, i) > 0.5 % remove large theta
            result_z(:, j, i) = [0, 0, 0]';
        end
    end
    data_z = squeeze(result_z(:,:,i))'; % P[kPa] theta[half] M[Nm] 
    csvtable_z = [csvtable_z data_z];
end
% 
% csvwrite('restructured_sm_dataset_y.csv', csvtable_y);
% csvwrite('restructured_sm_dataset_z.csv', csvtable_z);
% csvwrite('restructured_sm_dataset_x.csv', csvtable_x);
% design_parameters = squeeze(variable_value);
% design_parameters = [design_parameters(1,:); design_parameters(3:4,:)];
% csvwrite('design_parameters_sm.csv', design_parameters);
