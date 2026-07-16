
ALL_temp = t_TEMP_resize1;
ALL_temp_MEAN = mean(ALL_temp,4);
montage(ALL_temp_MEAN(:,:,300:319));

%%%

MAX =ALL_temp_MEAN(:,:,303);
img_av = MAX;

%%%

y_1 = floor(111/ 2);  % settare per ogni animale (br = bregma row, y)
x_2 = floor(129/ 2);  % settare per ogni animale (bc = bregma column, x)

img_av(y_1+45:y_1+55, x_2+15 : x_2+25, :) = 1;
img_av(y_1+45:y_1+55, x_2-25 : x_2-15, :) = 1;
img_av(y_1+42:y_1+52, x_2+59 : x_2+69, :) = 1;
img_av(y_1+42:y_1+52, x_2-69 : x_2-59, :) = 1;

valore_minimo = 0.3;
valore_massimo = 3;
%figure;
imagesc(img_av);
caxis([valore_minimo, valore_massimo]);
