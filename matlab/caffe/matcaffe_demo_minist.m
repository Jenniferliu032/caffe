function [scores, maxlabel] = matcaffe_demo_mnist(use_gpu)

% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init_minist(use_gpu);
else
  matcaffe_init_minist();
end
layers = caffe('get_weights');
%save layers_mnist_00.mat layers;
if nargin < 1 
  % For demo purposes we will use the peppers image
  im = imread('peppers.png');
end

% prepare oversampled input
% input_data is Height x Width x Channel x Num
[train_x,train_y,test_x,test_y] = prepare_image_minist;
% do forward pass to get scores
% scores are now Width x Height x Channels x Num
for n = 1:10
    errors = 0;
    fprintf('Processing the %dth iteration.\n',n);
    for m = 1:6000
        %disp(m)
        input_data = train_x(:,:,(m-1)*10+1:m*10);
        scores = caffe('forward', {reshape(input_data,[28,28,1,10])});
        %fprintf('Done with forward pass.\n');

        scores = scores{1};
        %save scores_mnist.mat scores;
        scores = squeeze(scores);

        % [~,maxlabel] = max(scores);

        % you can also get network weights by calling

        layers = caffe('get_weights');
        % fprintf('Done with get weights\n');
        % caffe('set_weights',layers);
        % fprintf('Done with set weights\n');
        save layers_minist.mat layers;

        y = train_y(:,(m-1)*10+1:m*10);
        save('scores_mnist.mat','scores','y');

        delta = single(scores - y);
        save delta_mnist.mat delta
        f = caffe('backward',{reshape(delta,[1,1,10,10])});
        %fprintf('Done with backward\n');
        %d = caffe('get_all_diff');
        %fprintf('Done with get diff\n');
        caffe('update');
        %fprintf('Done with update\n');
        %save diff_cifar.mat d;
    end
    for m = 1:1000
        input_data = test_x(:,:,(m-1)*10+1:m*10);
        scores = caffe('forward', {reshape(input_data,[28,28,1,10])});
        scores = scores{1};
        scores = squeeze(scores);
        [~,maxlabel] = max(scores);
        y = test_y(:,(m-1)*10+1:m*10);
        [~,tlabel] = max(y);
        errors = errors + length(find(maxlabel ~= tlabel));
    end
    rate = errors / 10000 ;
    disp(rate)
end
