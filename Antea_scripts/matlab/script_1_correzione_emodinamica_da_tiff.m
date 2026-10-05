%% (1) Hemodynamic correction, read directly from the raw interleaved TIFFs (no Fiji)
%
% Replaces "(1)_correzione_emodinamica.txt" for data that come off the camera as ONE
% folder per recording with both channels alternating (pv41_00001.tif, ...). Fiji
% used to split those into an emo stack and a gCaMP stack; load_interleaved_folder
% does that here. The correction is unchanged from script (1):
%
%   If2 = F_gcamp ./ mean_t(F_gcamp)          (each pixel over its own temporal mean)
%   Ir2 = F_emo   ./ mean_t(F_emo)
%   t_TEMP(:,:,:,K) = (If2 ./ Ir2 - 1) * 100  (dF/F in %, recording K)
%
% Afterwards run scripts (2) GSR, (3) ROI position and (4) correlation as before:
% they start from t_TEMP. Script (2) also needs a brain mask called Mask, on the
% same 256x256 grid as t_TEMP. Use script_2a_maschera_cervello.m to draw it here
% instead of in Fiji.
%
% Memory: t_TEMP is 256 x 256 x frames x recordings in double, about 1.6 GB per
% 3000-frame recording (7.9 GB for 5), plus about 5 GB while a recording is loaded.

%% parameters
% One animal-day: the folder holding the recording subfolders t1, t2, ...
animal_day_folder = 'C:\Users\loren\Downloads\WF_2026_new\260520\PV4';
recording_names = {'t1', 't2', 't3', 't4', 't5'};  % order of the 4th dimension of t_TEMP
channel_order = 'auto';   % 'auto' | 'gcamp_first' | 'emo_first' (see load_interleaved_folder)
scale = 0.5;              % imresize(out, 0.5, 'box') of script (1): 512x512 -> 256x256
n_trim = 0;               % frames dropped per channel; script (1) keeps all, Python drops 20

% Where load_interleaved_folder.m lives (this file's folder).
addpath(fileparts(mfilename('fullpath')));

%% correction, one recording at a time
n_recordings = numel(recording_names);
t_TEMP = [];
anatomy_image = [];   % mean reflectance of the first recording, for drawing the mask
for K = 1:n_recordings
    folder = fullfile(animal_day_folder, recording_names{K});
    [t_FILE_gCaMP, t_FILE_emo, info] = load_interleaved_folder(folder, ...
        'channel_order', channel_order, 'scale', scale, 'n_trim', n_trim);
    fprintf('%s: %d frames per channel; GCaMP starts at %s, emo at %s (levels odd %.0f / even %.0f)\n', ...
        recording_names{K}, info.n_frames, info.gcamp_first_file, info.emo_first_file, ...
        info.odd_level, info.even_level);

    MIf = mean(t_FILE_gCaMP, 3);
    MIr = mean(t_FILE_emo, 3);
    If2 = bsxfun(@rdivide, t_FILE_gCaMP, MIf);
    Ir2 = bsxfun(@rdivide, t_FILE_emo, MIr);
    t_temp = If2 ./ Ir2;

    % Allocate once the size is known; every recording must have the same length.
    if K == 1
        t_TEMP = zeros([size(t_temp), n_recordings]);
        anatomy_image = MIr;
    elseif size(t_temp, 3) ~= size(t_TEMP, 3)
        error('%s has %d frames, %s had %d: t_TEMP needs equal lengths.', ...
            recording_names{K}, size(t_temp, 3), recording_names{1}, size(t_TEMP, 3));
    end
    t_TEMP(:, :, :, K) = (t_temp - 1) * 100;   % DF/F in %

    clear t_FILE_gCaMP t_FILE_emo If2 Ir2 t_temp
end
fprintf('t_TEMP: %s\n', mat2str(size(t_TEMP)));
