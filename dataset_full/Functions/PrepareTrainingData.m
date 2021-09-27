function PrepareTrainingData()

global param;

sceneFolder = param.trainingScenes;
outputFolder = param.trainingData;

[~, scenePaths, numScenes] = GetFolderContent(sceneFolder, [], true);

endInd = 0;
MakeDir(outputFolder);
num_all = 0;

for i = 1 : numScenes
    
    count = fprintf('Started working on scene %d of %d \n', i, numScenes);
    
    %%% reading input data
    curExpo = ReadExpoTimes(scenePaths{i});
    [curImgsLDR, curLabel] = ReadImages(scenePaths{i});
    
    %%% processing data
    [inputs, label, num] = ComputeTrainingExamples(curImgsLDR, curExpo, curLabel);
    
    %%% writing data
    endInd = WriteTrainingExamples(inputs, label, endInd, [outputFolder, 'TrainSequence.h5']);
    
    fprintf(repmat('\b', [1, count]));
end

fprintf('Done\n\n');