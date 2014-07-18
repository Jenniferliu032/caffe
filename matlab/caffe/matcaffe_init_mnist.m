function  matcaffe_init_mnist(use_gpu, solver_def_file, model_file)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1 
  % By default use CPU
  use_gpu = 0;
end
if nargin < 2 || isempty(solver_def_file)
  % By default use imagenet_deploy
  solver_def_file = 'MNIST/mnist_solver.prototxt';
end
if nargin < 3 || isempty(model_file)
  % By default use caffe reference model
 % model_file = '../../examples/imagenet/caffe_reference_imagenet_model';
  model_file = '../../examples/cifar10/cifar10_full_iter_20000';
end

caffe('reset');

if caffe('is_initialized') == 0
  if exist(model_file, 'file') == 0
    % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
  end 
  if ~exist(solver_def_file,'file')
    % NOTE: you'll have to get network definition
    error('You need the solver prototxt definition');
  end 
  caffe('init', solver_def_file, model_file)
end
fprintf('Done with init\n');

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');

% put into test mode
caffe('set_phase_test');
fprintf('Done with set_phase_train\n');

caffe('presolve');
