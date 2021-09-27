function endInd = WriteTrainingExamples(inputs, label, endInd, savePath)

chunksz = 10;

startloc = [1, 1, 1, endInd+1];

SaveHDF(savePath, '/IN', single(inputs), PadWithOne(size(inputs), 4), startloc, chunksz, endInd == 0);
endInd = SaveHDF(savePath, '/GT', single(label) , PadWithOne(size(label), 4) , startloc, chunksz, endInd == 0);