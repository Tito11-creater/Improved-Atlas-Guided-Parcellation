function Improved_Atlas_Guided_Parcellation(Atlas,Data,Output_Name)
% IMPROVED_ATLAS_GUIDED_PARCELLATION Generates individualized brain atlas using IAGP framework
%
% Inputs:
%   Atlas       - File path to pre-defined brain atlas in Nifti format 
%                 (e.g., 'E:\data\Parcellation\Schaefer-400.nii')
%   Data        - File path to individual rs-fMRI data in Nifti format
%                 (e.g., 'E:\data\Data\rs-fMRI_sub001.nii')
%   Output_Name - File path to output individualized atlas in Nifti format
%                 (e.g., 'E:\data\Data\Schaefer-400-ind.nii')
%
% Note: 
%   Both Atlas and Data must have the same spatial resolution and be 
%   pre-registered to MNI space. The atlas should be a labeled volume where
%   non-zero values represent different ROIs.

% -------------------------------------------------------------------------
% Extract Atlas Information
% -------------------------------------------------------------------------

% fprintf('------------------\n');
% fprintf('Loading atlas data\n');
% fprintf('------------------\n');
[Pre_atlas,~,~,Header] = y_ReadAll(Atlas);
[dim1,dim2,dim3] = size(Pre_atlas);

% Find all non-zero voxels in the atlas
mask = Pre_atlas > 0;
n_voxel = sum(mask(:));
n_label = length(unique(Pre_atlas(mask)));

% Get coordinates and labels of all brain voxels
[i, j, k] = ind2sub([dim1, dim2, dim3], find(mask));
index_voxel = [i, j, k];
index_label = Pre_atlas(mask);

% Precompute adjacent voxel indices using 6-connectivity
index_adjacent = zeros(n_voxel, 6);
for n1 = 1:1:n_voxel  
    count = 1;     
    for n2 = 1:1:n_voxel
        d = sqrt((index_voxel(n1,1)-index_voxel(n2,1))^2+(index_voxel(n1,2)-index_voxel(n2,2))^2+(index_voxel(n1,3)-index_voxel(n2,3))^2);
        if d == 1   
            index_adjacent(n1,count) = n2;    % Each voxel has 6 adjacent voxels at most  
            count = count+1;
        end       
    end
end

% -------------------------------------------------------------------------
% Extract fMRI Data
% -------------------------------------------------------------------------

% fprintf('-----------------------------------\n');
% fprintf('Loading and preprocessing fMRI data\n');
% fprintf('-----------------------------------\n');

[BOLD_4D,~,~,~] = y_ReadAll(Data);
[~,~,~,TRs] = size(BOLD_4D);

BOLD = zeros(n_voxel,TRs);

for v = 1:n_voxel
    BOLD(v,:) = zscore(squeeze(BOLD_4D(index_voxel(v,1),index_voxel(v,2),index_voxel(v,3),:)));
end

% fprintf('----------------------------------------\n');
% fprintf('Computing functional connectivity matrix\n');
% fprintf('----------------------------------------\n');

FC = corr(BOLD');

% -------------------------------------------------------------------------
% Improved Atlas Guided Parcellation
% -------------------------------------------------------------------------

% fprintf('-----------------------\n');
% fprintf('Starting IAGP framework\n');
% fprintf('-----------------------\n');

% Step 1: Extract Functional Centers for each ROI
%
% The functional center is defined as the voxel within an ROI whose BOLD 
% time series has the highest correlation with the average BOLD time 
% series of that ROI.

% fprintf('--------------------------\n');
% fprintf('Finding functional centers\n');
% fprintf('--------------------------\n');
   
ROI_fun_center = zeros(n_label,1);  

label_mask = unique(Pre_atlas(mask));

for label = 1:n_label
    
    % Get voxels belonging to current ROI
    roi_mask = (index_label == label_mask(label));
    roi_bold = BOLD(roi_mask, :);
    
    % Compute mean BOLD time series for ROI
    BOLD_roi = mean(roi_bold, 1);
    
    % Find voxel with maximum correlation to ROI mean
    correlations = corr(roi_bold', BOLD_roi');
    [~, max_idx] = max(correlations);
    
    % Convert relative index to global index
    roi_indices = find(roi_mask);
    ROI_fun_center(label) = roi_indices(max_idx);
end

% Step 2: Region Growing with Threshold Constraints

% fprintf('--------------\n');
% fprintf('Region growing\n');
% fprintf('--------------\n');

% Initialize parameters
iter = 1;
r_threshold = 1;
r_step = 0.01;
iter_record = [];

index_label_ind = zeros(n_voxel,1);    % The individual atlas label of each voxel

% Initialize with functional centers
for label = 1:1:n_label         
    index_label_ind(ROI_fun_center(label)) = label_mask(label);
end

n_remain = n_voxel - n_label;

while n_remain > 0
    % Find candidate voxels for each ROI
    label_temp = zeros(n_label,2);

    for label = 1:n_label

        roi_voxels = find(index_label_ind == label_mask(label));

        % Find all unassigned adjacent voxels
        adjacent_candidates = [];
        for i = 1:length(roi_voxels)
            voxel_adj = index_adjacent(roi_voxels(i), :);
            voxel_adj = voxel_adj(voxel_adj > 0);
            unassigned_adj = voxel_adj(index_label_ind(voxel_adj) == 0);
            adjacent_candidates = [adjacent_candidates; unassigned_adj(:)];
        end

        adjacent_candidates = unique(adjacent_candidates);

        % Find candidate with highest mean connectivity to ROI
        mean_connections = zeros(length(adjacent_candidates), 1);
        for i = 1:length(adjacent_candidates)
            mean_connections(i) = mean(FC(adjacent_candidates(i), roi_voxels));
        end

        [best_corr, best_idx] = max(mean_connections);
        best_candidate = adjacent_candidates(best_idx);

        if best_candidate > 0
            label_temp(label, 1) = best_candidate;
            label_temp(label, 2) = best_corr;
        end
    end

    % Resolve overlapping assignments
    candidates = label_temp(:, 1);
    valid_candidates = candidates(candidates > 0);

    if length(valid_candidates) ~= length(unique(valid_candidates))
        % Resolve overlaps by assigning to ROI with highest correlation
        [unique_candidates, ~, ~] = unique(valid_candidates);
        for i = 1:length(unique_candidates)
            duplicate_indices = find(candidates == unique_candidates(i));
            if length(duplicate_indices) > 1
                [~, best_idx] = max(label_temp(duplicate_indices, 2));
                duplicate_indices(best_idx) = [];
                label_temp(duplicate_indices, 1) = 0;
            end
        end
    end

    % Assign voxels meeting threshold criteria
    assigned_this_iter = 0;
    for i = 1:n_label
        if label_temp(i, 1) > 0 && label_temp(i, 2) > r_threshold
            index_label_ind(label_temp(i, 1)) = label_mask(i);
            assigned_this_iter = assigned_this_iter + 1;
        end
    end

    % Update remaining voxels count
    n_remain = n_remain - assigned_this_iter;
    iter_record = [iter_record; n_remain];

    % Adjust threshold if no progress
    if iter > 1 && r_threshold > 0 && iter_record(iter) == iter_record(iter-1) && iter_record(iter) > 0
        r_threshold = r_threshold - r_step;	
    elseif iter > 1 && r_threshold < 0.01 && iter_record(iter) == iter_record(iter-1) && iter_record(iter) > 0
        unassigned = find(index_label_ind == 0);
        assigned = find(index_label_ind > 0);

        % Use Euclidean distance to find nearest labeled voxel
        for i = 1:length(unassigned)
            unassigned_pos = index_voxel(unassigned(i), :);

            % Calculate distances to all assigned voxels
            distances = sqrt(sum((index_voxel(assigned, :) - unassigned_pos).^2, 2));

            [~, min_idx] = min(distances);
            index_label_ind(unassigned(i)) = index_label_ind(assigned(min_idx));
        end
        n_remain = sum(index_label_ind == 0);
    end

    %fprintf('IAGP iter: %d, voxels remaining: %d, threshold: %.3f\n', iter, n_remain, r_threshold);
    iter = iter + 1;
       
end

% Reconstruct the individual atla in Nifti

% fprintf('--------------------------------------------\n');
% fprintf('Reconstruct individual atlas in Nifti format\n');
% fprintf('--------------------------------------------\n');

Atlas_individual = zeros(dim1,dim2,dim3);

for i = 1:1:n_voxel
    Atlas_individual(index_voxel(i, 1),index_voxel(i, 2),index_voxel(i, 3)) = index_label_ind(i);
end

y_Write(Atlas_individual,Header,Output_Name);

end
