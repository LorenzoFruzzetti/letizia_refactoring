function [volume, meta] = load_pixel_dump(recording_folder, volume_file)
%LOAD_PIXEL_DUMP  One recording of the Python pixel dumps, on Antea's 128x128 grid.
%
%   [volume, meta] = load_pixel_dump('E:\...\pixel_data\260716_PV3\t1', 'pixels_dff_full.npy')
%
%   The dumps (pixel_data\<day>_<animal>\<t#>\) hold the dF/F AFTER both 0.5 box
%   resizes, i.e. Antea's t_TEMP_resized from the start of script (2), but only a
%   76 x 87 crop around Bregma (rows -29..+47, cols -44..+43 from y_1/x_2 on the
%   128 grid; CLAUDE.md 9.15), as float16, and without the first 20 frames per
%   channel (the Python pipeline trims them; Antea's script (1) does not).
%
%   volume  128 x 128 x frames double, dF/F in %; NaN outside the crop. Script (2)
%           sets every pixel outside its mask to NaN anyway, so a mask inside the
%           crop gives the same result as the raw route. A mask reaching outside the
%           crop only uses its part inside the crop.
%   meta    struct: y_1, x_2 (Bregma on the 128 grid, Antea's variables),
%           bregma_row, bregma_col (256 grid), region (0-based [row0 row1 col0 col1),
%           the crop on the 128 grid), n_time, mean_f and mean_r (256 x 256 temporal
%           means of the two channels; mean_r is the anatomy image for drawing a mask).
%
%   volume_file:
%     'pixels_dff_full.npy'             mean-baseline dF/F: scripts (1)+(2)'s resize
%     'pixels_median_dff_20s_full.npy'  20 s running-median baseline instead

meta_dir = tempname;
mkdir(meta_dir);
remover = onCleanup(@() rmdir(meta_dir, 's'));
% An .npz is a zip of .npy files, one per key.
unzip(fullfile(recording_folder, 'pixels_meta_full.npz'), meta_dir);
for key = {'y_1', 'x_2', 'bregma_row', 'bregma_col', 'n_time'}
    meta.(key{1}) = double(read_npy(fullfile(meta_dir, [key{1} '.npy'])));
end
meta.region = double(read_npy(fullfile(meta_dir, 'region.npy')))';
grid = double(read_npy(fullfile(meta_dir, 'grid.npy')))';
meta.mean_f = read_npy(fullfile(meta_dir, 'mean_f.npy'));
meta.mean_r = read_npy(fullfile(meta_dir, 'mean_r.npy'));
meta.recording_folder = recording_folder;

crop = read_npy(fullfile(recording_folder, volume_file));   % frames x rows x cols
crop = permute(crop, [2 3 1]);                              % rows x cols x frames, as t_TEMP
expected = [meta.region(2) - meta.region(1), meta.region(4) - meta.region(3), meta.n_time];
if ~isequal(size(crop), expected)
    error('load_pixel_dump:shape', '%s: %s is %s, the metadata says %s.', recording_folder, ...
        volume_file, mat2str(size(crop)), mat2str(expected));
end

volume = NaN(grid(1), grid(2), size(crop, 3));
volume(meta.region(1) + 1 : meta.region(2), meta.region(3) + 1 : meta.region(4), :) = crop;
end
