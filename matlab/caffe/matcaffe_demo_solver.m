function [scores, maxlabel] = matcaffe_demo_solver(im, use_gpu)
% scores = matcaffe_demo(im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 


% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init_solver(use_gpu);
else
  matcaffe_init_solver();
end

if nargin < 1 
  % For demo purposes we will use the peppers image
  im = imread('peppers.png');
end

% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;input_data = {prepare_image_cifar(im)};
toc;save input_data.mat input_data;
fprintf('image size: %d, %d\n',size(input_data,1),size(input_data,2));
% do forward pass to get scores
% scores are now Width x Height x Channels x Num
%for m = 1:100

scores = caffe('forward', input_data);
fprintf('Done with forward pass.\n');

scores = scores{1};
scores = squeeze(scores);


[~,maxlabel] = max(scores);

% you can also get network weights by calling

layers = caffe('get_weights');
fprintf('Done with get weights\n');
caffe('set_weights',layers);
fprintf('Done with set weights\n');


for n = 1:length(maxlabel)
  y = zeros(size(scores));
    y(maxlabel(n),n)=1; 
end

delta = y - scores;
f = caffe('backward',{reshape(delta,[1,1,10,10])});
fprintf('Done with backward\n');
d = caffe('get_all_diff');
fprintf('Done with get diff\n');
caffe('update');
fprintf('Done with update\n');
