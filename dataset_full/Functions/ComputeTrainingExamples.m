function [inputs, label, num] = ComputeTrainingExamples(curImgsLDR, curExpo, curLabel)

global param;

patchSize = param.patchSize;
cropSize = param.cropSizeTraining;
stride = param.stride;
numAugment = param.numAugment;
numTotalAugment = param.numTotalAugment;

num = 1;
%%% prepare input features
curInputs = PrepareInputFeatures(curImgsLDR, curExpo);
inputs = curInputs;
label = curLabel;
% [height, width, depth] = size(curInputs);
% inputs = zeros(height, width, depth, num, 'single');
% inputs(:, :, :, 1) = curInputs;
% inputs(:, :, :, 2) = rot90(curInputs, 90);
% 
% label = zeros(height, width, 3, num, 'single');
% label(:, :, :, 1) = curLabel;
% label(:, :, :, 2) = rot90(curLabel, 90);



% inputs = GetPatches(curInputs, patchSize, stride);
% label = GetPatches(curLabel, patchSize, stride);
% 
% selInds = SelectSubset(inputs(:, :, 7:9, :));
% 
% inputs = inputs(:, :, :, selInds);
% label = label(:, :, :, selInds);
% num = length(selInds);


