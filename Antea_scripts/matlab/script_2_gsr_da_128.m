%% (2) Global signal regression, starting from t_TEMP_resized (128 x 128)
%
% Antea's "(2)_Global_Signal_Regression_SCRIPT.txt" without its first two steps:
% the uiopen of the Fiji mask (Mask now comes from script_2a_maschera_cervello.m) and
% the imresize of t_TEMP (script_1b_da_pixel_data.m already gives t_TEMP_resized at
% 128 x 128). Everything else is her code unchanged, including the global signal
% nanmean(nanmean(data,1),2) (the mean of the column means) and the per-pixel fitlm.
%
% Input:  t_TEMP_resized (128 x 128 x frames x recordings), Mask (256 x 256).
% Output: t_TEMP_regressed, as script (2).

Mask_resized = imresize(Mask,0.5,'box');

for i=1:size(Mask_resized,1)
    for j=1:size(Mask_resized,2)
if Mask_resized(i,j)==0
    t_TEMP_resized(i,j,:,:)=NaN;
end
    end
end


for i=1:size(t_TEMP_resized,4)
    data=t_TEMP_resized(:,:,:,i);
global_signal = squeeze(nanmean(nanmean(data,1),2));
data_regressed = zeros(size(data(:,:,:,1)));
for row=1:size(data,1)
    for col = 1:size(data,2)
if ~isnan(data(row,col,:))
    pixel_signal =squeeze(data(row,col,:));
mdl=fitlm(global_signal,pixel_signal);
global_component = mdl.Coefficients.Estimate(2)*global_signal+mdl.Coefficients.Estimate(1);
data_regressed(row,col,:)=pixel_signal-global_component;
else
data_regressed(row,col,:)=NaN;
end
    end
end
t_TEMP_regressed(:,:,:,i)=data_regressed;
fprintf('GSR recording %d of %d done\n', i, size(t_TEMP_resized,4));
end
