function roi_set = read_roi_set(yaml_path)
%READ_ROI_SET  Read one ROI set written by the Python ROI editor (roi_sets/rebuilt/*.yaml).
%
%   roi_set = read_roi_set('E:\...\roi_sets\rebuilt\260520_PV4_t1.yaml')
%
%   Returns a struct:
%     name                     e.g. '260520_PV4_t1'
%     bregma_row, bregma_col   Bregma on the 256x256 (once-downsampled) grid
%     y_1, x_2                 Bregma on the final 128x128 grid, floor(bregma/2): the
%                              y_1 / x_2 of Antea's scripts (3) and (4)
%     labels                   {1 x 22} box names in file order
%     boxes                    [22 x 4] row_start, row_end, col_start, col_end,
%                              offsets from (y_1, x_2), both ends inclusive
%
%   A box covers img(y_1+row_start : y_1+row_end, x_2+col_start : x_2+col_end, :),
%   exactly the indexing of Antea's script (4). The baseline atlas
%   (roi_sets/cortex22_roi_set.yaml) holds her offsets unchanged; the rebuilt sets are
%   the same constellation rescaled per animal by its Bregma-Lambda distance.
%
%   MATLAB has no YAML reader, so this parses the fixed one-line layout the Python
%   tool writes ('  BFDR: {row_start: 17, row_end: 22, col_start: 37, col_end: 42}')
%   and raises if a file does not look like that.

text = fileread(yaml_path);
roi_set.name = regexp(text, '(?m)^name:\s*(\S+)', 'tokens', 'once');
roi_set.name = roi_set.name{1};
roi_set.bregma_row = read_integer(text, 'bregma_row', yaml_path);
roi_set.bregma_col = read_integer(text, 'bregma_col', yaml_path);
roi_set.y_1 = floor(roi_set.bregma_row / 2);
roi_set.x_2 = floor(roi_set.bregma_col / 2);

grid = regexp(text, '(?m)^grid:\s*\[(\d+),\s*(\d+)\]', 'tokens', 'once');
if isempty(grid) || ~isequal(str2double(grid), [128 128])
    error('read_roi_set:grid', '%s: expected grid [128, 128].', yaml_path);
end

tokens = regexp(text, ['(?m)^\s+(\w+):\s*\{row_start:\s*(-?\d+),\s*row_end:\s*(-?\d+),' ...
    '\s*col_start:\s*(-?\d+),\s*col_end:\s*(-?\d+)\}'], 'tokens');
if isempty(tokens)
    error('read_roi_set:boxes', '%s: no boxes in the expected one-line format.', yaml_path);
end
roi_set.labels = cellfun(@(t) t{1}, tokens, 'UniformOutput', false);
roi_set.boxes = cell2mat(cellfun(@(t) str2double(t(2:5)), tokens, 'UniformOutput', false)');
end


function value = read_integer(text, key, yaml_path)
token = regexp(text, ['(?m)^' key ':\s*(-?\d+)'], 'tokens', 'once');
if isempty(token)
    error('read_roi_set:key', '%s: no %s.', yaml_path, key);
end
value = str2double(token{1});
end
