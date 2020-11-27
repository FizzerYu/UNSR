function bicubic()
clear all; close all; clc

image_path  = '/titan_data2/lichangyu/sr_dataset/LR/LRBI/Urban100/x4';   % 全路径
image_name = {'img_076.png', 'img_096.png'};    % 文件名

scale = 4;

for idx_set = 1:length(image_name)
    
    im_LR = imread(fullfile(image_path, image_name{idx_set}));
    
    im_SR = imresize(im_LR, scale, 'bicubic');

    folder_LR = fullfile('./upsample_bicubic');
    if ~exist(folder_LR)
        mkdir(folder_LR)
    end
    fprintf('%d. %s: ', idx_set, image_name{idx_set}(1:end-4));

    fn_SR = fullfile(folder_LR, [image_name{idx_set}(1:end-4), 'upsample_by_bicubic', '.png']);
    
    imwrite(im_SR, fn_SR, 'png');
end  
end 