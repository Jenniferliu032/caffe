function images = prepare_image(im)
% ------------------------------------------------------------------------
d = load('ilsvrc_2012_mean');
[~/liusifei/caffe_parser/matlab/caffe/matcaffe_demo_solver.m][matlab]                [Line:98/130,Column:1][75%]
I0716 15:24:48.609802 25985 net.cpp:125] Top shape: 10 32 16 16 (81920)
function images = prepare_image_cifar(im)
% ------------------------------------------------------------------------
% d = load('ilsvrc_2012_mean');
% IMAGE_MEAN = d.image_mean;
IMAGE_DIM = min(size(im,1),size(im,2));
CROPPED_DIM = 32; 

% resize to fixed input size
im = single(im);
% im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]);

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
curr = 1;
for i = indices
  for j = indices
    images(:, :, :, curr) = ... 
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end 
end
center = floor(indices(2) / 2)+1;
images(:,:,:,5) = ... 
    permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ... 
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);
