clearvars; clearvars -global; clc; close all;

%% settings
addpath(genpath('Functions'));

InitParam();

global param;

param.trainingScenes = './Data/Training/';
param.trainingData = './Train/';
[~, param.trainingNames, ~] = GetFolderContent(param.trainingData, '.h5');

param.testScenes = './Data/Test/';
param.testData = './Test/';
[~, param.testNames, ~] = GetFolderContent(param.testData, '.h5');

fprintf('***************************\n');
fprintf('Preparing the training data\n');
fprintf('***************************\n\n');
PrepareTrainingData();

fprintf('***************************\n');
fprintf('Preparing the test data\n');
fprintf('***************************\n\n');
PrepareTestData();
