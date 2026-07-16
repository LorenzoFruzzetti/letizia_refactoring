
%% lettura del file
matfiles = (dir(fullfile('/media/NASini/OPTOGEN/250401/GTA13/t2')));
nfiles = length(matfiles);
sizet=0;
K=1

for i= 5:4:size(matfiles,1)
name=strcat(matfiles(i).folder,{'/'},matfiles(i+1).name)
NAME=name{1,1};
%copyfile(NAME, '~/temp.tif')
%NAME =  '~/temp.tif';
tiff_info = imfinfo(NAME);
sizet=0;
out = [];
%%% ricorda che se pesa piÙ di 4gb non lo legge!!
for ii = 1:size(tiff_info, 1)
out(:,:,sizet + ii) = imread(NAME, ii);
end
t_FILE{1} = out(:,:,21:end);
t_FILE_emo = imresize(t_FILE{1},0.5,'box');
sizet =  sizet + size(tiff_info, 1);
ii
name=strcat(matfiles(i).folder,{'/'},matfiles(i).name)
NAME=name{1,1};
%copyfile(NAME, '~/temp.tif')
%NAME =  '~/temp.tif';
tiff_info = imfinfo(NAME);
sizet=0;
out = [];
%%% ricorda che se pesa piÙ di 4gb non lo legge!!
for ii = 1:size(tiff_info, 1)
out(:,:,sizet + ii) = imread(NAME, ii);
end
t_FILE{2} = out(:,:,21:end);
t_FILE_gCaMP = imresize(t_FILE{2},0.5,'box');
sizet =  sizet + size(tiff_info, 1);

%%%% stimolato
MIf = mean(t_FILE_gCaMP(:,:,1:278),3);
MIr = mean(t_FILE_emo(:,:,1:278),3);


If2 = bsxfun(@rdivide, t_FILE_gCaMP, MIf);
Ir2 = bsxfun(@rdivide, t_FILE_emo, MIr);
t_temp = If2./Ir2;
t_TEMP(:,:,:,K) = (((t_temp-1)*100)); % ottieni il DF/F in %;
K=K+1;
end



for i=1:size(t_TEMP,4)
    temp = (t_TEMP(:,:,:,i));
    temp_resized = imresize(temp,0.5,'box');
    t_TEMP_resize1(:,:,:,i)=temp_resized;
end

