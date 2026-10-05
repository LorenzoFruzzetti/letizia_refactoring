%% (1b) Start from the Python pixel dumps (pixel_data) instead of the raw TIFFs
%
% For sessions whose raw TIFFs are not on this machine. pixel_data\<day>_<animal>\<t#>\
% holds the corrected dF/F after BOTH 0.5 resizes, so this builds Antea's
% t_TEMP_resized (128 x 128 x frames x recordings) directly: the state at the end
% of the first loop of script (2). Then:
%   script_2a_maschera_cervello.m   Mask (drawn on mean_r, or 'roi_union')
%   script_2_gsr_da_128.m           script (2) from t_TEMP_resized on
%   script_4_correlazione_roi_set.m ROI traces and correlation
%
% Differences from the raw route (script_1_correzione_emodinamica_da_tiff.m):
% - only the 76 x 87 crop around Bregma exists; outside it the frames are NaN. A
%   'roi_union' mask lies inside the crop, so its GSR is unaffected. A hand-drawn
%   brain outline reaching outside the crop uses only its part inside.
% - the first 20 frames of each channel are missing (2980 frames, not 3000), as the
%   Python pipeline trims them. The raw route matches with n_trim = 20.
% - dF/F is stored as float16: rounding up to ~0.008 percentage points, which moves
%   the correlations by about 1e-5.

%% parameters
pixel_data_dir = 'E:\Developing_projects\letizia\pixel_data';
day = '260716';
animal = 'PV3';
recording_names = {'t1', 't2', 't3', 't4', 't5'};  % order of the 4th dimension
volume_file = 'pixels_dff_full.npy';   % mean baseline, as Antea; 'pixels_median_dff_20s_full.npy' = 20 s median

addpath(fileparts(mfilename('fullpath')));

%% load
n_recordings = numel(recording_names);
t_TEMP_resized = [];
pixel_meta = cell(1, n_recordings);
for K = 1:n_recordings
    folder = fullfile(pixel_data_dir, [day '_' animal], recording_names{K});
    [volume, meta] = load_pixel_dump(folder, volume_file);
    pixel_meta{K} = meta;
    fprintf('%s: %d frames, crop rows %d-%d cols %d-%d of 128, y_1 = %d, x_2 = %d\n', ...
        recording_names{K}, size(volume, 3), meta.region(1) + 1, meta.region(2), ...
        meta.region(3) + 1, meta.region(4), meta.y_1, meta.x_2);
    if K == 1
        t_TEMP_resized = zeros([size(volume), n_recordings]);
        anatomy_image = meta.mean_r;   % 256 x 256 mean reflectance, for script 2a
    elseif size(volume, 3) ~= size(t_TEMP_resized, 3)
        error('%s has %d frames, %s had %d.', recording_names{K}, size(volume, 3), ...
            recording_names{1}, size(t_TEMP_resized, 3));
    end
    t_TEMP_resized(:, :, :, K) = volume;
end
clear volume meta
fprintf('t_TEMP_resized: %s\n', mat2str(size(t_TEMP_resized)));
