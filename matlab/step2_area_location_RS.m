
    img_av = (t_TEMP_resize1(:,:,302,1));

%ALL_temp = t_TEMP_resize1;
%ALL_temp_MEAN = mean(ALL_temp,4);
%montage(t_TEMP_resize1(:,:,300:319));
%MAX =ALL_temp_MEAN(:,:,304);
%img_av = MAX;

%%

y_1 = floor(121/ 2);  % settare per ognai animale (br = bregma row, y)
x_2 = floor(134/ 2);  % settare per ogni animale (bc = bregma column, x)

img_av(y_1+22:y_1+27, x_2+0 : x_2+5, :) = 1;
img_av(y_1+22:y_1+27, x_2-12 : x_2-7, :) = 1;
img_av(y_1+21:y_1+26, x_2+29 : x_2+34, :) = 1;
img_av(y_1+21:y_1+26, x_2-37 : x_2-32, :) = 1;

valore_minimo = 0.3;
valore_massimo = 3;
%figure;
imagesc(img_av);
caxis([valore_minimo, valore_massimo]);
