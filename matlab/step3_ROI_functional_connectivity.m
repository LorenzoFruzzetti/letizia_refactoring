TEMP_ROI=[];
img_av = [];

for i=1:size(t_TEMP_resize1,4)
img_av = t_TEMP_resize1(:,:,:,i);
% img_av = Y(:,:,:);

  % settare per ogni animale (br = bregma row, y)
  % settare per ogni animale (bc = bregma column, x)

Verme_R_bis = img_av(y_1+22:y_1+27, x_2+0 : x_2+5, :) ;
Verme_R = nanmean(nanmean(Verme_R_bis,1),2);
Verme_R = Verme_R(:,:);

Verme_L_bis = img_av(y_1+22:y_1+27, x_2-12 : x_2-7, :) ;
Verme_L = nanmean(nanmean(Verme_L_bis,1),2);
Verme_L = Verme_L(:,:);

Laterale_R_bis = img_av(y_1+21:y_1+26, x_2+29 : x_2+34, :) ;
Laterale_R = nanmean(nanmean(Laterale_R_bis,1),2);
Laterale_R = Laterale_R(:,:);

Laterale_L_bis = img_av(y_1+21:y_1+26, x_2-34 : x_2-29, :) ;
Laterale_L = nanmean(nanmean(Laterale_L_bis,1),2);
Laterale_L = Laterale_L(:,:);

regioni_R = cat(2,Laterale_R',Verme_R');
regioni_L = cat(2,Laterale_L',Verme_L');
ALL = cat(2,regioni_L,regioni_R);
TEMP_ROI(:,:,i) = ALL;
end
%%
R=[];
for i=1:size(TEMP_ROI,3)
R(:,:,i)=corr(TEMP_ROI(:,:,i),TEMP_ROI(:,:,i));
end
 
averaged_traces = mean(TEMP_ROI,3);
R_mean = mean(R,3);
