clear
clc

scale_factor = 16;%图像在计算的时候的缩放大小8 12  32(大图)  2或3（小图到大图）    5.12 大图12 8不行   16

stack_img_dir = 'E:\Zhu\myownex\ETreg\img\TOM\stack\result\';
ava_img_dir = 'E:\Zhu\myownex\ETreg\img\TOM\stack\ava\';

file_name = dir(stack_img_dir);
num_stack = length(file_name) - 2;

%% 求stack的均值
for i = 1:num_stack
    img_dir = [stack_img_dir, file_name(i + 2).name, '\'];
    img_name = dir([img_dir,'*.tif']);
    num_img = length(img_name);
    imgava = 0.0;
    for j = 0:num_img-1
        imgtemp = imread([img_dir,num2str(j,'%03d'),'.tif']);
        imgava = imgava + double(imgtemp);
    end
    imgava = imgava / num_img;
    imwrite(uint8(imgava),[ava_img_dir,num2str(i-1,'%03d'),'.tif']);
end

%% 进行插补图像数目最少（简单）估计
outlier_filter = fspecial('gaussian',[5,5],1);
smooth_filter = fspecial('gaussian',[18,18],8);   % 256 有四分之一一起动，才是比较典型的动 25 11  18

% insert_imgnum_int = [];  %a(end+1)=5
insert_imgnum_int = zeros(1, num_stack - 1);
for i = 1:num_stack-1
    img1ava = imread([ava_img_dir,num2str(i - 1,'%03d'),'.tif']);
    img1ava=imresize(img1ava, 1/scale_factor, 'bicubic');
    
    img2ava = imread([ava_img_dir,num2str(i,'%03d'),'.tif']);
    img2ava=imresize(img2ava, 1/scale_factor, 'bicubic');
    
    difimg_ava = img2ava - img1ava;  % 体块像素漂移数目
    difimg_ava = imfilter(difimg_ava,outlier_filter,'replicate');
    difimg_ava = imfilter(difimg_ava,smooth_filter,'replicate');
    
    if i == 1
        [scale_height_img,scale_width_img] = size(img1ava);
    end
    
    img1_dir = [stack_img_dir, file_name(i + 2).name, '\'];
    img1_name = dir([img1_dir,'*.tif']);
    num_img1 = length(img1_name);
    img1 = imread([img1_dir,num2str(num_img1 - 1,'%03d'),'.tif']);  % 从后往前计算
    img1 = imresize(img1, 1/scale_factor, 'bicubic');
    img_pixlnum_map1 = zeros(scale_height_img, scale_width_img);
    for j = num_img1 - 1 : (-1): 1
        img1temp = imread([img1_dir,num2str(j - 1,'%03d'),'.tif']);
        img1temp = imresize(img1temp, 1/scale_factor, 'bicubic');
        difimg = img1temp - img1;
        difimg = imfilter(difimg,smooth_filter,'replicate');
        img_num_onepixl = (num_img1 - j) ./ double(difimg);  % 一个像素变几张图
        img_num_onepixl(~isfinite(img_num_onepixl))=0;  % 替换inf和nan 为0
        imgmap_weitht = (img_pixlnum_map1 == 0);   % 找未变化的像素位置
        difimg = img_num_onepixl .* imgmap_weitht; % 这次统计的是上次未变化的
        img_pixlnum_map1 = img_pixlnum_map1 + difimg;
        if sum(sum(img_pixlnum_map1 ~= 0)) == scale_height_img * scale_width_img   % img_pixlnum_map + img_scale_mask 是否可以用做mask 从而减少sift的提取量，mask内的点权重增大，抖动移到外面，（不变的点是不是枝干点，最需要配到一起）
            break;
        end
    end
    img_pixlnum_map1 = imfilter(img_pixlnum_map1,smooth_filter,'replicate');
    
    img2_dir = [stack_img_dir, file_name(i + 3).name, '\'];
    img2_name = dir([img2_dir,'*.tif']);
    num_img2 = length(img2_name);
    img2 = imread([img1_dir,num2str(0,'%03d'),'.tif']);  % 从前往后计算
    img2 = imresize(img2, 1/scale_factor, 'bicubic');
    img_pixlnum_map2 = zeros(scale_height_img, scale_width_img);
    for j = 2 : num_img2
        img2temp = imread([img2_dir,num2str(j - 1,'%03d'),'.tif']);
        img2temp = imresize(img2temp, 1/scale_factor, 'bicubic');
        difimg = img2temp - img2;
        difimg = imfilter(difimg,smooth_filter,'replicate');
        img_num_onepixl = (j-1) ./ double(difimg);  % 一个像素变几张图
        img_num_onepixl(~isfinite(img_num_onepixl))=0;
        imgmap_weitht = (img_pixlnum_map2 == 0);
        difimg = img_num_onepixl .* imgmap_weitht;
        img_pixlnum_map2 = img_pixlnum_map2 + difimg;
        if sum(sum(img_pixlnum_map2 ~= 0)) == scale_height_img * scale_width_img   % img_pixlnum_map + img_scale_mask 是否可以用做mask 从而减少sift的提取量，mask内的点权重增大，抖动移到外面，（不变的点是不是枝干点，最需要配到一起）
            break; 
        end
    end
    img_pixlnum_map2 = imfilter(img_pixlnum_map2,smooth_filter,'replicate');
    
    img_pixlnum_map = (img_pixlnum_map1 + img_pixlnum_map2) / 2;
    
    insert_imgnum_map = double(difimg_ava) .* img_pixlnum_map;
    insert_imgnum = max(max(insert_imgnum_map));
    insert_imgnum_int(i) = round(insert_imgnum);
end

%% QCQP求解每段的具体数目
% 每层上下分别求解
H = zeros(20,20);
H(1,1) = 2;
H(20,19) = -4;
H(20,20) = 2;
for m=2:19
    H(m,m-1) = -4;
    H(m,m) = 4;
end
Aeq = zeros(19,20);
for m=1:19
    Aeq(m,m:m+1) = 1;
end
stackloss = [16, 45, 66, 52, 53, 58, 40, 50, 51, 58];
stackid = 1;
insertid = 1;
beq = zeros(19,1);
for m=1:19
    if mod(m,2) ~= 0
        beq(m,1) = stackloss(stackid);
        stackid = stackid + 1;
    else
        beq(m,1) = insert_imgnum_int(insertid);
        insertid = insertid + 1;
    end
end
% A = ones(20,20);
A = diag(ones(20,1),0);
A = A * (-1);
b = zeros(20,1);
[x,fval] = quadprog(H,[],A,b,Aeq,beq);
x = round(x);   % [43;58;60;45]  10 6 27 18 27 39 29 23 39 14 33 25 23 17 19 31 3 48 25 33    33 45 68 62 47 48 36 34 73 
