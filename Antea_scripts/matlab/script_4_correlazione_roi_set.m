%% (3)+(4) ROI traces and correlation, with the ROI sets built by the Python ROI editor
%
% Replaces "(3)_FOV128-128_ROIposition.txt" and "(4)_FOV128x128_corr_SCRIPT.txt".
% Instead of one hand-set y_1 / x_2 and Antea's fixed offsets, every recording uses
% its own ROI set, roi_sets/rebuilt/<day>_<animal>_<t#>.yaml: its own Bregma and the
% same 22-box constellation rescaled to that animal (read_roi_set.m). The rest is
% script (4) unchanged: nanmean inside each box, columns in the order
% M2_alta M2_bassa M1_alta M1_bassa BFD Tr FL HL RS_alta V1a V1 (left, then right),
% Pearson correlation per recording, mean over recordings.
%
% Input:  t_TEMP_regressed from script (2) (128 x 128 x frames x recordings).
%         Without GSR, set t_TEMP_regressed = t_TEMP_resized first.
% Output: the variables of script (4)
%         R_regressed_ANIMAL_XDPL       22 x 22 x recordings, Pearson r
%         mean_R_regressed_ANIMAL_XDPL  22 x 22, mean over recordings
%         TEMP_regressed_ANIMAL_XDPL    frames x 22 x recordings, ROI traces
%         roi_labels                    {1 x 22}, the column order
% and a figure of the boxes on one frame (what script (3) drew).

%% parameters
roi_set_dir = 'E:\Developing_projects\letizia\roi_sets\rebuilt';
day = '260520';
animal = 'PV4';
recording_names = {'t1', 't2', 't3', 't4', 't5'};  % same order as the 4th dimension of t_TEMP_regressed
overlay_frame = 11;            % frame shown under the boxes (script (3) used 11)

% Antea's column order (script (4): regioni_L then regioni_R).
region_order = {'M2L_alta', 'M2L_bassa', 'M1L_alta', 'M1L_bassa', 'BFDL', 'TrL', 'FLL', 'HLL', ...
    'RSL_alta', 'V1aL', 'V1L', 'M2R_alta', 'M2R_bassa', 'M1R_alta', 'M1R_bassa', 'BFDR', 'TrR', ...
    'FLR', 'HLR', 'RSR_alta', 'V1aR', 'V1R'};

addpath(fileparts(mfilename('fullpath')));

%% ROI traces per recording
n_recordings = size(t_TEMP_regressed, 4);
if n_recordings ~= numel(recording_names)
    error('t_TEMP_regressed has %d recordings but recording_names lists %d.', ...
        n_recordings, numel(recording_names));
end
n_frames = size(t_TEMP_regressed, 3);
n_regions = numel(region_order);
TEMP = zeros(n_frames, n_regions, n_recordings);   % frames x 22 x recordings
roi_sets = cell(1, n_recordings);
for i_rec = 1:n_recordings
    roi_set = read_roi_set(fullfile(roi_set_dir, sprintf('%s_%s_%s.yaml', day, animal, recording_names{i_rec})));
    roi_sets{i_rec} = roi_set;
    fprintf('%s: Bregma %d/%d -> y_1 = %d, x_2 = %d\n', roi_set.name, roi_set.bregma_row, ...
        roi_set.bregma_col, roi_set.y_1, roi_set.x_2);
    img_av = t_TEMP_regressed(:, :, :, i_rec);
    for i_region = 1:n_regions
        i_box = find(strcmp(roi_set.labels, region_order{i_region}));
        if numel(i_box) ~= 1
            error('%s has no box %s.', roi_set.name, region_order{i_region});
        end
        box = roi_set.boxes(i_box, :);   % row_start row_end col_start col_end
        rows = roi_set.y_1 + (box(1):box(2));
        cols = roi_set.x_2 + (box(3):box(4));
        if rows(1) < 1 || cols(1) < 1 || rows(end) > size(img_av, 1) || cols(end) > size(img_av, 2)
            error('%s: box %s falls outside the %dx%d frame.', roi_set.name, ...
                region_order{i_region}, size(img_av, 1), size(img_av, 2));
        end
        box_pixels = img_av(rows, cols, :);
        TEMP(:, i_region, i_rec) = squeeze(nanmean(nanmean(box_pixels, 1), 2));
    end
end
roi_labels = region_order;

%% correlation, as script (4)
R = zeros(n_regions, n_regions, n_recordings);
for i_rec = 1:n_recordings
    R(:, :, i_rec) = corr(TEMP(:, :, i_rec), TEMP(:, :, i_rec));
end
R_regressed_ANIMAL_XDPL = R;                  % Pearson r per recording
mean_R_regressed_ANIMAL_XDPL = mean(R, 3);    % mean over the recordings of this animal-day
TEMP_regressed_ANIMAL_XDPL = TEMP;            % ROI traces, frames x 22 x recordings

%% boxes on one frame of the first recording (what script (3) showed)
roi_set = roi_sets{1};
img_overlay = t_TEMP_regressed(:, :, overlay_frame, 1);
figure('Name', sprintf('%s ROI boxes', roi_set.name));
imagesc(img_overlay);
axis image;
colormap(gca, gray);
hold on;
for i_box = 1:size(roi_set.boxes, 1)
    box = roi_set.boxes(i_box, :);
    % Pixel centres are integers, so a box spans [start-0.5, end+0.5].
    rectangle('Position', [roi_set.x_2 + box(3) - 0.5, roi_set.y_1 + box(1) - 0.5, ...
        box(4) - box(3) + 1, box(2) - box(1) + 1], 'EdgeColor', 'y');
    text(roi_set.x_2 + box(3), roi_set.y_1 + box(1) - 1.5, roi_set.labels{i_box}, ...
        'Color', 'y', 'FontSize', 6, 'Interpreter', 'none');   % keep "_" in M2L_alta literal
end
plot(roi_set.x_2, roi_set.y_1, 'r+', 'MarkerSize', 10);
title(sprintf('%s, frame %d, Bregma (y_1 = %d, x_2 = %d)', strrep(roi_set.name, '_', '\_'), ...
    overlay_frame, roi_set.y_1, roi_set.x_2));
