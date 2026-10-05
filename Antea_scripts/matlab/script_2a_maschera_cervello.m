%% (2a) Brain mask for script (2), drawn in MATLAB instead of Fiji
%
% Script (2) starts with uiopen(...) of a mask TIFF made in Fiji and then uses a
% variable called Mask on the 256x256 grid of t_TEMP (it resizes it by 0.5 itself).
% Run this after script_1_correzione_emodinamica_da_tiff.m, which leaves
% anatomy_image (mean reflectance of the first recording, 256x256), then skip the
% uiopen line of script (2).
%
% mask_source:
%   'draw'  click the outline of the cortex on anatomy_image; double-click to close
%   'file'  read an existing mask image (e.g. one already made in Fiji); any
%           non-zero pixel is inside
%   'roi_union'  the union of the 22 ROI boxes of a Python ROI set
%           (roi_sets/rebuilt/<day>_<animal>_<t#>.yaml). This is the mask our Python
%           GSR uses (roi_pixel_connectivity.py --gsr, the "_gsr" variants), so the
%           global signal is then the same. It covers only the boxes, not the whole
%           cortex: a smaller mask than a hand-drawn brain outline.

%% parameters
mask_source = 'draw';
mask_file = '';             % for 'file': path to the mask image
roi_set_file = '';          % for 'roi_union': e.g. 'E:\Developing_projects\letizia\roi_sets\rebuilt\260520_PV4_t1.yaml'
save_mask_to = '';          % optional: save Mask as a .tif here ('' = do not save)

%% mask
switch mask_source
    case 'draw'
        figure('Name', 'Draw the brain mask');
        imagesc(anatomy_image);
        axis image;
        colormap gray;
        title('Click around the cortex, double-click to close');
        outline = drawpolygon();
        Mask = createMask(outline);
    case 'file'
        Mask = imread(mask_file) > 0;
    case 'roi_union'
        % Union of the boxes on the final 128x128 grid, then each pixel doubled to
        % 2x2, so script (2)'s imresize(Mask, 0.5, 'box') gives the union back exactly.
        addpath(fileparts(mfilename('fullpath')));
        roi_set = read_roi_set(roi_set_file);
        union_128 = false(128, 128);
        for i_box = 1:size(roi_set.boxes, 1)
            box = roi_set.boxes(i_box, :);
            union_128(roi_set.y_1 + (box(1):box(2)), roi_set.x_2 + (box(3):box(4))) = true;
        end
        Mask = kron(union_128, true(2));
    otherwise
        error('mask_source must be ''draw'', ''file'' or ''roi_union'', not ''%s''.', mask_source);
end
% Mask lives on the 256x256 grid in both routes (raw TIFFs: t_TEMP; pixel_data:
% anatomy_image = mean_r), and script (2) resizes it to 128.
if ~isequal(size(Mask), [256 256])
    error('Mask is %s; script (2) expects 256x256.', mat2str(size(Mask)));
end
Mask = double(Mask);   % script (2) tests Mask_resized(i,j)==0 after a box resize
fprintf('Mask: %d of %d pixels inside\n', nnz(Mask), numel(Mask));
if ~isempty(save_mask_to)
    imwrite(uint8(Mask * 255), save_mask_to);
end
