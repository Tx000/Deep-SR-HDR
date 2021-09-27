function InitParam()

global param;
global gamma;
global mu;

param.testNet = 'TrainedNetwork';
gamma = 2.2;
mu = 5000;

%%% If you have compiled MatConvNet with GPU and CuDNN supports, then leave
%%% these parameters as is. Otherwise change them appropriately.
param.useGPU = true; 
param.gpuMethod = 'Cudnn';%'NoCudnn';%


%%%%%%%%%%%%%%%%%% Training Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.patchSize = 256;
param.stride = 64;
param.cropSizeTraining = 0; % we crop the boundaries to avoid artifacts in the training
param.batchSize = 20;
param.numAugment = 10; % number of randomly selected data augmentation
param.numTotalAugment = 48; % number of total data augmentation combinations
param.border = 0;
param.weps = 1e-6;

param.trainingScenes = 'F:/HDR/SIG17_HDR_Code_v1.0/TrainingData/Training/';
param.trainingData = 'TrainingData/';
[~, param.trainingNames, ~] = GetFolderContent(param.trainingData, '.h5');

param.testScenes = 'TrainingData/Test/';
param.testData = 'TrainingData/Test/';
[~, param.testNames, ~] = GetFolderContent(param.testData, '.h5');

param.trainNet = 'TrainingData';


param.continue = true;
param.startIter = 0;

param.testNetIter = 5000;
param.printInfoIter = 5;


%%% ADAM parameters
param.alpha = 0.0001;
param.beta1 = 0.9;
param.beta2 = 0.999;
param.eps = 1e-8;















