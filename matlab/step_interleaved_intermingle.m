% Interleaved ("intermingle") modality — MATLAB reference.
%
% Mirrors the Python `--layout interleaved` path (wfci.io.load_interleaved_folder
% + resting-state pipeline) so the two languages can be cross-checked on the same
% sample data. The other three layouts (stack / folder / stream) consume the same
% two channels this script splits out, so validating interleaved validates them
% all against MATLAB.
%
% What it does (identical math to gen_reference.m, single interleaved trial):
%   1. read the sorted single-page TIFFs in data/ (both channels, alternating);
%   2. split odd-positioned images (1st,3rd,5th,7th) from even-positioned
%      (2nd,4th,6th); the brighter group (mean of its top-10% pixels, first image
%      of each group) becomes gcamp, the dimmer emo; truncate to equal length;
%   3. step 1 — trim(=0) -> imresize 0.5 'box' -> DFF -> imresize 0.5 'box';
%   4. step 3 — 4 ROI nanmean traces -> per-trial corr -> R_mean.
%
% Parameters match the Python benchmark: trim=0, ds=0.5, y_1=60, x_2=67, RS
% (full-recording baseline and correlation window).
%
% Run (paths via env vars, with defaults relative to this script):
%   set WFCI_DATA=...\data & set WFCI_INTERLEAVED_OUT=...\interleaved_reference.mat
%   matlab -batch "run('matlab/step_interleaved_intermingle.m')"

here = fileparts(mfilename('fullpath'));

data_dir = getenv('WFCI_DATA');
if isempty(data_dir)
    data_dir = fullfile(here, '..', 'data');
end
out_file = getenv('WFCI_INTERLEAVED_OUT');
if isempty(out_file)
    out_file = fullfile(here, '..', 'tests', 'matlab_reference', 'interleaved_reference.mat');
end

%% --- read all single-page frames (sorted by name) -----------------------
listing = dir(fullfile(data_dir, '*.tif'));
names = sort({listing.name});
raw = [];
for k = 1:numel(names)
    raw(:, :, k) = double(imread(fullfile(data_dir, names{k})));   %#ok<AGROW>
end

%% --- split odd/even and pick the brighter group as gcamp ----------------
odd_idx = 1:2:numel(names);    % 1st, 3rd, 5th, 7th
even_idx = 2:2:numel(names);   % 2nd, 4th, 6th
odd = raw(:, :, odd_idx);
even = raw(:, :, even_idx);

% Brightness proxy: mean of the top-10% pixels of each group's first image
% (matches wfci.io._top_percent_mean; the odd/even margin is large, so the
% decision is robust regardless of the exact percentile convention).
top10_mean = @(img) mean_of_top_percent(img, 10);
if top10_mean(odd(:, :, 1)) >= top10_mean(even(:, :, 1))
    gcamp_raw = odd;  emo_raw = even;
else
    gcamp_raw = even; emo_raw = odd;
end

% Keep channels the same length (odd frame count -> groups differ by one).
n = min(size(gcamp_raw, 3), size(emo_raw, 3));
gcamp_raw = gcamp_raw(:, :, 1:n);
emo_raw = emo_raw(:, :, 1:n);

%% --- fixed pipeline parameters (identical to the Python benchmark) -------
trim = 0;
ds = 0.5;
y_1 = 60;
x_2 = 67;
% ROI boxes: order [Laterale_L, Verme_L, Laterale_R, Verme_R].
box_rows = [21 26; 22 27; 21 26; 22 27];
box_cols = [-34 -29; -12 -7; 29 34; 0 5];

%% --- step 1: trim -> resize -> DFF -> resize (single interleaved trial) --
g = gcamp_raw(:, :, trim+1:end);
r = emo_raw(:, :, trim+1:end);
g = imresize(g, ds, 'box');       % 512 -> 256
r = imresize(r, ds, 'box');
MIf = mean(g, 3);                 % resting-state: full-recording baseline
MIr = mean(r, 3);
If2 = bsxfun(@rdivide, g, MIf);
Ir2 = bsxfun(@rdivide, r, MIr);
dff = (If2 ./ Ir2 - 1) * 100;
dff_stack = imresize(dff, ds, 'box');   % 256 -> 128, [y, x, time]

%% --- step 3: ROI traces + connectivity ----------------------------------
n_time = size(dff_stack, 3);
TEMP_ROI = zeros(n_time, 4);
for j = 1:4
    rr = (y_1 + box_rows(j, 1)) : (y_1 + box_rows(j, 2));
    cc = (x_2 + box_cols(j, 1)) : (x_2 + box_cols(j, 2));
    patch = dff_stack(rr, cc, :);
    v = nanmean(nanmean(patch, 1), 2);
    TEMP_ROI(:, j) = v(:);
end

R = corr(TEMP_ROI, TEMP_ROI);           % full-recording correlation
R_mean = R;                             % single trial -> mean over trials == R
averaged_traces = TEMP_ROI;

%% --- save for the Python comparison -------------------------------------
save(out_file, 'TEMP_ROI', 'R', 'R_mean', 'averaged_traces', ...
     'y_1', 'x_2', 'trim', 'ds', '-v7');
fprintf('Wrote %s\n', out_file);

% --- local helper: mean of the brightest top_percent%% of an image --------
function m = mean_of_top_percent(img, top_percent)
    v = sort(double(img(:)), 'descend');
    k = max(1, ceil(numel(v) * top_percent / 100));
    m = mean(v(1:k));
end
