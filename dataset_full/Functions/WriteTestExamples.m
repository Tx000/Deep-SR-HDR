function WriteTestExamples(inputs, label, savePath)

chunksz = 10;

startloc = [1, 1, 1, 1];

SaveHDF(savePath, '/IN', single(inputs), PadWithOne(size(inputs), 4), startloc, chunksz, true);
SaveHDF(savePath, '/GT', single(label) , PadWithOne(size(label), 4) , startloc, chunksz, true);