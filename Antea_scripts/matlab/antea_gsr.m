function data_regressed = antea_gsr(data, mask)
%ANTEA_GSR  Antea's global signal regression (script (2)) on one recording.
%
%   data_regressed = antea_gsr(data, mask)
%
%   data   rows x cols x frames dF/F of ONE recording on the 128x128 grid
%          (t_TEMP_resized(:,:,:,i) in script (2)).
%   mask   rows x cols, nonzero = brain. Antea's Mask_resized; for the ROI-union
%          mask this is the union of the 22 boxes on the 128 grid.
%
%   The loop of "(2)_Global_Signal_Regression_SCRIPT.txt" unchanged, as a function so a
%   driver can call it per recording: pixels outside the mask become NaN, the global
%   signal is nanmean(nanmean(data,1),2) (the mean of the column means), and every
%   pixel with no NaN frame is replaced by its residual from fitlm(global_signal, pixel).

for i = 1:size(mask, 1)
    for j = 1:size(mask, 2)
        if mask(i, j) == 0
            data(i, j, :) = NaN;
        end
    end
end

global_signal = squeeze(nanmean(nanmean(data, 1), 2));
data_regressed = zeros(size(data(:, :, :, 1)));
for row = 1:size(data, 1)
    for col = 1:size(data, 2)
        if ~isnan(data(row, col, :))
            pixel_signal = squeeze(data(row, col, :));
            mdl = fitlm(global_signal, pixel_signal);
            global_component = mdl.Coefficients.Estimate(2) * global_signal + mdl.Coefficients.Estimate(1);
            data_regressed(row, col, :) = pixel_signal - global_component;
        else
            data_regressed(row, col, :) = NaN;
        end
    end
end
end
