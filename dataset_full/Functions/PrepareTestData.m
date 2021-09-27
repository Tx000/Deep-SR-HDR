function PrepareTestData()

global param;

sceneFolder = param.testScenes;
outputFolder = param.testData;

[sceneNames, scenePaths, numScenes] = GetFolderContent(sceneFolder, [], true);

MakeDir(outputFolder);

for i = 1 : numScenes
    
    count = fprintf('Started working on scene %d of %d', i, numScenes);
    
    %%% reading input data
    curExpo = ReadExpoTimes(scenePaths{i});
    [curImgsLDR, curLabel] = ReadImages(scenePaths{i});
    
    %%% processing data
    inputs = ComputeTestExamples(curImgsLDR, curExpo);
    
    %%% writing data
    WriteTestExamples(inputs, curLabel, [outputFolder, '/', sceneNames{i}, '.h5']);
    
    fprintf(repmat('\b', [1, count]));
end

fprintf('Done\n\n');