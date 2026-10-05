function [gcamp, emo, info] = load_interleaved_folder(folder, varargin)
%LOAD_INTERLEAVED_FOLDER  Read one recording's raw interleaved TIFFs into two stacks.
%
%   Replaces the Fiji step before script (1): the camera writes both channels,
%   alternating, into ONE folder of single-frame TIFFs (pv41_00001.tif,
%   pv41_00002.tif, ...). This splits them into GCaMP and reflectance (emo) and
%   downsamples each frame as it is read.
%
%   [gcamp, emo, info] = load_interleaved_folder(folder)
%   [gcamp, emo, info] = load_interleaved_folder(folder, 'channel_order', 'emo_first', ...)
%
%   Options (name/value):
%     'channel_order'  'auto' (default) | 'gcamp_first' | 'emo_first'
%                      Sorted by name, odd files are one channel and even files the
%                      other. 'auto' calls GCaMP the group whose FIRST image is dimmer
%                      (mean of its brightest 10% of pixels): on this rig reflectance
%                      is several times brighter. Same rule as the Python pipeline
%                      (wfci.io.interleaved_channel_files). Getting it wrong swaps the
%                      channels and inverts the correction, so check info.
%     'scale'          0.5 (default). Each frame goes through
%                      imresize(frame, scale, 'box'). A box resize acts on each frame
%                      separately, so this is the same as Antea's
%                      imresize(out, 0.5, 'box') on the whole stack, but needs a
%                      quarter of the memory (512x512x3000 in double is 6.3 GB per
%                      channel, 256x256x3000 is 1.6 GB).
%     'n_trim'         0 (default). Frames dropped from the start of EACH channel.
%                      Antea's script (1) keeps every frame; the Python pipeline
%                      (profile cerebellar_rs) drops 20.
%     'pattern'        '*.tif' (default).
%     'top_percent'    10 (default). Brightest-pixel percentage used by 'auto'.
%
%   Outputs:
%     gcamp, emo  [rows, cols, frames] double, the same as t_FILE_gCaMP / t_FILE_emo
%                 in script (1).
%     info        struct: which files went to which channel, the two brightness
%                 levels 'auto' compared, and the frame count.

p = inputParser;
addRequired(p, 'folder', @(x) ischar(x) || isstring(x));
addParameter(p, 'channel_order', 'auto', @(x) any(strcmp(x, {'auto', 'gcamp_first', 'emo_first'})));
addParameter(p, 'scale', 0.5, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'n_trim', 0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'pattern', '*.tif');
addParameter(p, 'top_percent', 10, @(x) isnumeric(x) && x > 0 && x <= 100);
parse(p, folder, varargin{:});
opt = p.Results;
folder = char(folder);

% dir() does not promise any order, so sort by name explicitly. The frame index
% in the file names is zero-padded, so name order is acquisition order.
files = dir(fullfile(folder, opt.pattern));
if numel(files) < 2
    error('load_interleaved_folder:noFiles', ...
        'Need at least 2 files matching %s in %s, found %d.', opt.pattern, folder, numel(files));
end
names = sort({files.name});
odd_names = names(1:2:end);     % 1st, 3rd, 5th ... file
even_names = names(2:2:end);    % 2nd, 4th, 6th ... file

% Which group is GCaMP.
odd_level = NaN;
even_level = NaN;
switch opt.channel_order
    case 'gcamp_first'
        gcamp_names = odd_names;
        emo_names = even_names;
    case 'emo_first'
        gcamp_names = even_names;
        emo_names = odd_names;
    case 'auto'
        odd_level = top_percent_mean(imread(fullfile(folder, odd_names{1})), opt.top_percent);
        even_level = top_percent_mean(imread(fullfile(folder, even_names{1})), opt.top_percent);
        if odd_level <= even_level
            gcamp_names = odd_names;
            emo_names = even_names;
        else
            gcamp_names = even_names;
            emo_names = odd_names;
        end
end

% Equal length (an odd file count leaves one group a frame longer), then trim.
n_frames = min(numel(gcamp_names), numel(emo_names));
if opt.n_trim >= n_frames
    error('load_interleaved_folder:trim', 'n_trim=%d leaves no frames (%d per channel).', ...
        opt.n_trim, n_frames);
end
gcamp_names = gcamp_names(1 + opt.n_trim : n_frames);
emo_names = emo_names(1 + opt.n_trim : n_frames);

gcamp = read_stack(folder, gcamp_names, opt.scale);
emo = read_stack(folder, emo_names, opt.scale);

info = struct( ...
    'folder', folder, ...
    'channel_order', opt.channel_order, ...
    'gcamp_first_file', gcamp_names{1}, ...
    'emo_first_file', emo_names{1}, ...
    'odd_level', odd_level, ...
    'even_level', even_level, ...
    'n_frames', numel(gcamp_names), ...
    'n_trim', opt.n_trim, ...
    'scale', opt.scale);
end


function stack = read_stack(folder, names, scale)
% Read and downsample every frame into a preallocated [rows, cols, frames] stack.
first = imresize(double(imread(fullfile(folder, names{1}))), scale, 'box');
stack = zeros(size(first, 1), size(first, 2), numel(names));
stack(:, :, 1) = first;
for i_frame = 2:numel(names)
    stack(:, :, i_frame) = imresize(double(imread(fullfile(folder, names{i_frame}))), scale, 'box');
end
end


function level = top_percent_mean(image, top_percent)
% Mean of the brightest top_percent% of pixels: compares the two channels on the
% illuminated cortex rather than on the dark background. Used only to decide which
% group is dimmer, and the two channels differ several-fold, so the exact
% percentile definition does not matter.
values = sort(double(image(:)), 'descend');
n_top = max(1, round(numel(values) * top_percent / 100));
level = mean(values(1:n_top));
end
