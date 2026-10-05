%% PV3 / PV4 / PV5: Antea's matrices and difference figures for every day, all in MATLAB
%
% For every session of each animal: load the 5 recordings from pixel_data
% (load_pixel_dump.m, = Antea's t_TEMP_resized), optionally run her GSR (antea_gsr.m)
% with the recording's ROI-union mask, take the box means of the recording's own ROI
% set (read_roi_set.m), correlate (corr), and average the recordings: mean_R =
% mean(R,3), as script (4). Then, as script (5): the mean over animals, and
% day N - day 1 differences, drawn with imagesc / parula / her Allen labels.
%
% Outputs (output_dir):
%   matrices_<tag>.png      rows PV3, PV4, PV5, mean; columns day 1..5: mean_R
%   difference_<tag>.png    same rows; columns day 2..5 minus day 1
%   mean_R_<tag>.mat        mean_R (22 x 22 x animals x days), sessions, labels
% <tag> is the volume name plus _gsr or _no_gsr.
%
% Run (MATLAB): run this file. Batch: matlab -batch "run('...\make_pv_figures.m')".
% About 8 minutes with GSR (fitlm on 792 mask pixels x 75 recordings), 2 without.

%% parameters
repo = 'E:\Developing_projects\letizia';
pixel_data_dir = fullfile(repo, 'pixel_data');
roi_set_dir = fullfile(repo, 'roi_sets', 'rebuilt');
output_dir = fullfile(repo, 'outputs', 'pv_matlab_figures');
animals = {'PV3', 'PV4', 'PV5'};
baseline_day_index = 1;
volume_file = 'pixels_dff_full.npy';   % mean-baseline dF/F, as Antea's script (1)
gsr_runs = [true, false];              % with GSR (Antea's chain) and without
color_limits = [-1 1];                 % fixed scale of every panel (Pearson r / difference)
recompute = false;   % false: reuse mean_R_<tag>.mat when it exists (only redraw); true: always recompute

% Column order of script (4) and the labels of script (5) for it.
region_order = {'M2L_alta', 'M2L_bassa', 'M1L_alta', 'M1L_bassa', 'BFDL', 'TrL', 'FLL', 'HLL', ...
    'RSL_alta', 'V1aL', 'V1L', 'M2R_alta', 'M2R_bassa', 'M1R_alta', 'M1R_bassa', 'BFDR', 'TrR', ...
    'FLR', 'HLR', 'RSR_alta', 'V1aR', 'V1R'};
allen_labels = {'MOs-a_L','MOs-p_L','MOp-a_L','MOp-p_L','SSp-bfd_L','SSp-tr_L','SSp-fL_L', ...
    'SSp-hl_L','RSP_L','VISa_L','VISp_L','MOs-a_R','MOs-p_R','MOp-a_R','MOp-p_R','SSp-bfd_R', ...
    'SSp-tr_R','SSp-fL_R','SSp-hl_R','RSP_R','VISa_R','VISp_R'};

addpath(fileparts(mfilename('fullpath')));
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% sessions of each animal, from the experimental design
design = readtable(fullfile(pixel_data_dir, 'experimental_design.csv'), 'TextType', 'string');
design.day = string(design.day);
n_animals = numel(animals);
day_indices = unique(design.day_index(ismember(design.animal, animals)))';
n_days = numel(day_indices);
n_regions = numel(region_order);

for use_gsr = gsr_runs
    tag = erase(volume_file, '.npy');
    if use_gsr
        tag = [tag '_gsr'];
    else
        tag = [tag '_no_gsr'];
    end
    fprintf('=== %s\n', tag);
    started = tic;

    %% mean_R per animal-day: 22 x 22 x animals x days
    mat_file = fullfile(output_dir, ['mean_R_' tag '.mat']);
    if ~recompute && isfile(mat_file)
        saved = load(mat_file);
        if ~isequal(saved.animals, animals) || ~isequal(saved.day_indices, day_indices) ...
                || ~strcmp(saved.volume_file, volume_file) || saved.use_gsr ~= use_gsr
            error('%s was made with other settings; set recompute = true.', mat_file);
        end
        mean_R = saved.mean_R;
        sessions = saved.sessions;
        fprintf('reusing %s\n', mat_file);
    else
    mean_R = NaN(n_regions, n_regions, n_animals, n_days);
    sessions = strings(n_animals, n_days);
    for i_animal = 1:n_animals
        for i_day = 1:n_days
            rows = design(design.animal == animals{i_animal} & design.day_index == day_indices(i_day), :);
            if isempty(rows)
                error('%s has no session with day_index %d.', animals{i_animal}, day_indices(i_day));
            end
            day = rows.day(1);
            recordings = sort(rows.recording);
            sessions(i_animal, i_day) = day + "_" + animals{i_animal};
            R = zeros(n_regions, n_regions, numel(recordings));
            for k = 1:numel(recordings)
                folder = fullfile(pixel_data_dir, char(sessions(i_animal, i_day)), char(recordings(k)));
                [volume, meta] = load_pixel_dump(folder, volume_file);
                roi_set = read_roi_set(fullfile(roi_set_dir, sprintf('%s_%s_%s.yaml', day, ...
                    animals{i_animal}, recordings(k))));
                if roi_set.y_1 ~= meta.y_1 || roi_set.x_2 ~= meta.x_2
                    error('%s: ROI set Bregma (%d, %d) differs from the dump (%d, %d).', folder, ...
                        roi_set.y_1, roi_set.x_2, meta.y_1, meta.x_2);
                end
                if use_gsr
                    volume = antea_gsr(volume, roi_union_mask(roi_set, size(volume, 1), size(volume, 2)));
                end
                traces = roi_traces(volume, roi_set, region_order);   % frames x 22
                R(:, :, k) = corr(traces, traces);
            end
            mean_R(:, :, i_animal, i_day) = mean(R, 3);
            fprintf('%s: %d recordings (%.0f s elapsed)\n', sessions(i_animal, i_day), ...
                numel(recordings), toc(started));
        end
    end
    save(mat_file, 'mean_R', 'sessions', 'animals', ...
        'day_indices', 'region_order', 'allen_labels', 'volume_file', 'use_gsr');
    end

    %% figure 1: mean_R per day (+ mean over animals, as script (5))
    row_names = [animals, {'mean'}];
    matrices = cell(n_animals + 1, n_days);
    titles = cell(n_animals + 1, n_days);
    for i_day = 1:n_days
        for i_animal = 1:n_animals
            matrices{i_animal, i_day} = mean_R(:, :, i_animal, i_day);
            titles{i_animal, i_day} = sprintf('%s (day %d)', sessions(i_animal, i_day), day_indices(i_day));
        end
        matrices{end, i_day} = mean(mean_R(:, :, :, i_day), 3);
        titles{end, i_day} = sprintf('mean %s: day %d', strjoin(animals, '+'), day_indices(i_day));
    end
    draw_grid(matrices, titles, allen_labels, color_limits, 'Pearson r', ...
        sprintf('%s: mean R per day (MATLAB: mean(R,3) over 5 recordings)', tag), ...
        fullfile(output_dir, ['matrices_' tag '.png']));

    %% figure 2: DIFF = day N - day 1 (script (5)'s DIFF, groups replaced by days)
    i_base = find(day_indices == baseline_day_index);
    later = setdiff(1:n_days, i_base);
    differences = cell(n_animals + 1, numel(later));
    titles = cell(n_animals + 1, numel(later));
    for i_col = 1:numel(later)
        i_day = later(i_col);
        for i_animal = 1:n_animals
            differences{i_animal, i_col} = mean_R(:, :, i_animal, i_day) - mean_R(:, :, i_animal, i_base);
            titles{i_animal, i_col} = sprintf('%s: %s - %s', row_names{i_animal}, ...
                sessions(i_animal, i_day), sessions(i_animal, i_base));
        end
        differences{end, i_col} = mean(mean_R(:, :, :, i_day), 3) - mean(mean_R(:, :, :, i_base), 3);
        titles{end, i_col} = sprintf('mean %s: day %d - day %d', strjoin(animals, '+'), ...
            day_indices(i_day), baseline_day_index);
    end
    draw_grid(differences, titles, allen_labels, color_limits, 'difference in Pearson r', ...
        sprintf('%s: DIFF = mean R day N - mean R day %d (MATLAB)', tag, baseline_day_index), ...
        fullfile(output_dir, ['difference_' tag '.png']));
    fprintf('%s done in %.1f min\n', tag, toc(started) / 60);
end


%% local functions
function mask = roi_union_mask(roi_set, n_rows, n_cols)
% Union of the 22 boxes on the 128 grid: Antea's Mask_resized for script_2a's 'roi_union'.
mask = zeros(n_rows, n_cols);
for i_box = 1:size(roi_set.boxes, 1)
    box = roi_set.boxes(i_box, :);
    mask(roi_set.y_1 + (box(1):box(2)), roi_set.x_2 + (box(3):box(4))) = 1;
end
end


function traces = roi_traces(volume, roi_set, region_order)
% frames x regions box means, script (4)'s nanmean(nanmean(box,1),2), in region_order.
traces = zeros(size(volume, 3), numel(region_order));
for i_region = 1:numel(region_order)
    i_box = find(strcmp(roi_set.labels, region_order{i_region}));
    if numel(i_box) ~= 1
        error('%s has no box %s.', roi_set.name, region_order{i_region});
    end
    box = roi_set.boxes(i_box, :);
    box_pixels = volume(roi_set.y_1 + (box(1):box(2)), roi_set.x_2 + (box(3):box(4)), :);
    traces(:, i_region) = squeeze(nanmean(nanmean(box_pixels, 1), 2));
end
end


function draw_grid(matrices, titles, labels, color_limits, colorbar_label, figure_title, file_path)
% One imagesc panel per cell of `matrices`, parula, fixed colour limits, Antea's labels.
[n_rows, n_cols] = size(matrices);
% Size set on the page, not the screen: a batch figure is clipped to the screen size,
% which shrank every panel. print() renders at the page size instead.
page_size = [3.9 * n_cols, 3.6 * n_rows];   % inches
fig = figure('Visible', 'off', 'PaperUnits', 'inches', 'PaperPosition', [0 0 page_size], ...
    'PaperSize', page_size);
layout = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
for i_row = 1:n_rows
    for i_col = 1:n_cols
        ax = nexttile(layout);
        imagesc(ax, matrices{i_row, i_col});
        colormap(ax, parula(256));
        clim(ax, color_limits);
        axis(ax, 'square');
        xticks(ax, 1:numel(labels));
        yticks(ax, 1:numel(labels));
        xticklabels(ax, labels);
        yticklabels(ax, labels);
        xtickangle(ax, 90);
        ax.TickLabelInterpreter = 'none';
        ax.FontSize = 5;
        title(ax, titles{i_row, i_col}, 'FontSize', 8, 'Interpreter', 'none');
        bar = colorbar(ax);
        bar.Label.String = colorbar_label;
        bar.FontSize = 6;
    end
end
title(layout, figure_title, 'Interpreter', 'none', 'FontSize', 11);
print(fig, file_path, '-dpng', '-r200');
close(fig);
end
