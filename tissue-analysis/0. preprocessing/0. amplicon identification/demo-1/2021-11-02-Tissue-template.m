%% Hu RIBOmap tissue test

run_id = 'max_cluster';

input_path = '/stanley/WangLab/Data/Processed/2021-11-23-Hu-MouseBrainRIBOmap/';

addpath('/stanley/WangLab/Documents/starmap_colab/Pipeline/');
addpath('/stanley/WangLab/Documents/starmap_colab/Code/matlab/');
addpath('/stanley/WangLab/Documents/starmap_colab/Code/matlab/myfunction/');

input_dim = [2048 2048 35 4 9]; % ignore 5th channel

useGPU = false;

% data_dirs = dir(fullfile(input_path, "round1", "Position*"));
% data_dirs = struct2table(data_dirs);
% data_dirs = natsort(data_dirs.name);
%%
% disp('haha')

% d = gpuDevice(1);

% Ndirs = numel(data_dirs);
% curr_data_dir = data_dirs{p};

curr_data_dir = tile;
curr_out_path = fullfile(input_path, 'output', run_id, curr_data_dir);

if ~exist(curr_out_path, 'dir')
    mkdir(curr_out_path)
end

curr_img_out_path = fullfile(input_path, 'output', run_id, 'dots_image');

if ~exist(curr_img_out_path, 'dir')
    mkdir(curr_img_out_path)
end

sdata_start = tic;
sdata = new_STARMapDataset(input_path, 'useGPU', useGPU);
sdata.log = fopen(fullfile(curr_out_path, 'log.txt'), 'w');
fprintf(sdata.log, sprintf("====Current Position: %s====\n", curr_data_dir));
sdata = sdata.LoadRawImages('sub_dir', curr_data_dir, 'input_dim', input_dim);
sdata = sdata.SwapChannels;
sdata = sdata.MinMaxNormalize;
sdata = sdata.HistEqualize('Method', "inter_round");
sdata = sdata.HistEqualize('Method', "intra_round");
sdata = sdata.MorphoRecon('Method', "2d", 'radius', 6);
sdata = sdata.test_GlobalRegistration('useGPU', false);
sdata = sdata.LocalRegistration('Iterations', 50, 'AccumulatedFieldSmoothing', 1);
sdata = sdata.LoadCodebook('remove_index', 5);
sdata = sdata.SpotFinding('Method', "max3d", 'ref_index', 1, 'showPlots', false);
sdata = sdata.ReadsExtraction('voxelSize', [1 1 1]);
sdata = sdata.ReadsFiltration('mode', "duo", 'endBases', ['C', 'T'], 'split_loc', 5, 'showPlots', false);

%     % Save registered image
%     output_dir = fullfile(curr_out_path, 'registered_image');
%     if ~exist(output_dir, 'dir')
%        mkdir(output_dir);
%     end
%     new_SaveImg(output_dir, sdata.registeredImages);

% Save round 1 image
output_dir = fullfile(input_path, 'output', run_id, 'round1_merged');
if ~exist(output_dir, 'dir')
   mkdir(output_dir);
end
r1_img = max(sdata.registeredImages(:,:,:,:,1), [], 4);
r1_img_name = fullfile(output_dir, sprintf("%s.tif", curr_data_dir));
SaveSingleTiff(r1_img, r1_img_name)

% Save points (allSpots)
allSpots = sdata.allSpots;
allReads = sdata.allReads;

save(fullfile(curr_out_path, strcat('allPoints_max3d.mat')), 'allReads', 'allSpots');


% Save points (goodSpots)
goodSpots = sdata.goodSpots;
goodReads = sdata.goodReads;

save(fullfile(curr_out_path, strcat('goodPoints_max3d.mat')), 'goodReads', 'goodSpots');

fprintf(sprintf("====%s Finished [time=%02f]====", curr_data_dir, toc(sdata_start)));
fprintf(sdata.log, sprintf("====%s Finished [time=%02f]====", curr_data_dir, toc(sdata_start)));

% Save dots image
curr_img = max(sdata.registeredImages(:,:,:,:,1), [], 4);
curr_img = max(curr_img, [], 3);

if ~isempty(goodSpots)
    plot_centroids(goodSpots, curr_img, 2, 'r')
    saveas(gcf, fullfile(curr_img_out_path, sprintf("%s.tif", curr_data_dir)));
end

fclose(sdata.log);
