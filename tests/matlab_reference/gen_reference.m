% Generate MATLAB reference outputs for the Python parity test.
%
% Reads the real sample frames in ../../data (7 single-page 512x512 TIFFs =
% 7 time frames of one channel), builds a deterministic 2-trial, two-channel
% dataset from them, then runs the SAME math as the Python `wfci` package using
% MATLAB's native imresize('box'), nanmean and corr. Both the exact inputs and
% all outputs are saved to reference.mat so the Python test consumes identical
% inputs and compares against MATLAB's own functions.
%
% Run:  matlab -batch "run('tests/matlab_reference/gen_reference.m')"

here = fileparts(mfilename('fullpath'));
data_dir = fullfile(here, '..', '..', 'data');

%% --- read sample frames (sorted) into a 512x512x7 double stack ---
listing = dir(fullfile(data_dir, 'R11_*.tif'));
names = sort({listing.name});               % deterministic filename order
raw = [];
for k = 1:numel(names)
    img = double(imread(fullfile(data_dir, names{k})));
    raw(:, :, k) = img;                      %#ok<AGROW>
end

%% --- deterministic two-channel, two-trial construction (strictly positive) ---
gcamp = {}; emo = {};
gcamp{1} = raw + 50;          emo{1} = 0.5 * raw + 200;
gcamp{2} = 1.1 * raw + 30;    emo{2} = 0.4 * raw + 150;
n_trial = numel(gcamp);

%% --- fixed pipeline parameters (kept identical in the Python test) ---
trim = 0;         % sample has only 7 frames; no 20-frame trim here
ds   = 0.5;       % box downsample factor
y_1  = 60;        % downsampled Bregma row  (fits the 128x128 grid)
x_2  = 67;        % downsampled Bregma col

% ROI boxes: MATLAB 1-based inclusive offsets from (y_1, x_2), matching
% wfci.config.ROIConfig defaults. Order = [Laterale_L, Verme_L, Laterale_R, Verme_R].
box_rows = [21 26; 22 27; 21 26; 22 27];     % [row_start row_end] offsets
box_cols = [-34 -29; -12 -7; 29 34; 0 5];    % [col_start col_end] offsets

% Two analysis modes (windows are MATLAB 1-based inclusive):
%   RS   : baseline = full, corr = full
%   STIM : baseline = 1:4,  corr = 2:6
modes = struct( ...
    'name',    {'rs', 'stim'}, ...
    'base_lo', {1, 1}, 'base_hi', {NaN, 4}, ...   % NaN hi = full recording
    'corr_lo', {1, 2}, 'corr_hi', {NaN, 6});      % NaN hi = full recording

%% --- helper: full step-1 for one trial (trim -> resize -> DFF -> resize) ---
% (inline, mirrors wfci.correction.build_dff_stack)

results = struct();
for m = 1:numel(modes)
    md = modes(m);

    dff_stack = [];   % [y, x, time, trial]
    for t = 1:n_trial
        g = gcamp{t}(:, :, trim+1:end);
        r = emo{t}(:, :, trim+1:end);
        g = imresize(g, ds, 'box');           % 512 -> 256
        r = imresize(r, ds, 'box');
        nt = size(g, 3);
        if isnan(md.base_hi), bhi = nt; else, bhi = md.base_hi; end
        MIf = mean(g(:, :, md.base_lo:bhi), 3);
        MIr = mean(r(:, :, md.base_lo:bhi), 3);
        If2 = bsxfun(@rdivide, g, MIf);
        Ir2 = bsxfun(@rdivide, r, MIr);
        dff = (If2 ./ Ir2 - 1) * 100;
        dff = imresize(dff, ds, 'box');        % 256 -> 128
        dff_stack(:, :, :, t) = dff;           %#ok<AGROW>
    end

    % --- step 3: ROI time-series + connectivity ---
    n_time = size(dff_stack, 3);
    TEMP_ROI = zeros(n_time, 4, n_trial);
    for t = 1:n_trial
        img = dff_stack(:, :, :, t);
        for j = 1:4
            rr = (y_1 + box_rows(j, 1)) : (y_1 + box_rows(j, 2));
            cc = (x_2 + box_cols(j, 1)) : (x_2 + box_cols(j, 2));
            patch = img(rr, cc, :);
            v = nanmean(nanmean(patch, 1), 2);
            TEMP_ROI(:, j, t) = v(:);
        end
    end

    if isnan(md.corr_hi), chi = n_time; else, chi = md.corr_hi; end
    R = zeros(4, 4, n_trial);
    for t = 1:n_trial
        R(:, :, t) = corr(TEMP_ROI(md.corr_lo:chi, :, t), TEMP_ROI(md.corr_lo:chi, :, t));
    end
    R_mean = mean(R, 3);
    averaged_traces = mean(TEMP_ROI, 3);

    results.(md.name).dff_stack = dff_stack;
    results.(md.name).TEMP_ROI = TEMP_ROI;
    results.(md.name).R = R;
    results.(md.name).R_mean = R_mean;
    results.(md.name).averaged_traces = averaged_traces;
    results.(md.name).base_lo = md.base_lo;
    results.(md.name).base_hi = md.base_hi;   % NaN means "full"
    results.(md.name).corr_lo = md.corr_lo;
    results.(md.name).corr_hi = md.corr_hi;
end

%% --- direct imresize('box') checks incl. ODD dimensions (general algorithm) ---
resize_cases = struct('in', {}, 'out', {});
mats = { double(reshape(1:9, 3, 3)), ...
         double(reshape(1:25, 5, 5)), ...
         double(reshape(1:35, 7, 5)), ...
         double(raw(1:31, 1:29, 1)) };     % odd x odd real-data crop
for c = 1:numel(mats)
    resize_cases(c).in = mats{c};
    resize_cases(c).out = imresize(mats{c}, ds, 'box');
end

%% --- save everything the Python test needs ---
trials_gcamp = cat(4, gcamp{:});   % [512,512,7,ntrial]
trials_emo   = cat(4, emo{:});
params = struct('trim', trim, 'ds', ds, 'y_1', y_1, 'x_2', x_2, ...
                'box_rows', box_rows, 'box_cols', box_cols);

out_file = fullfile(here, 'reference.mat');
save(out_file, 'trials_gcamp', 'trials_emo', 'params', 'results', ...
     'resize_cases', '-v7');
fprintf('Wrote %s\n', out_file);
