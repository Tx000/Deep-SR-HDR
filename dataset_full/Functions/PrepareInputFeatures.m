function curInputs = PrepareInputFeatures(curImgsLDR, curExpo)

numImgs = length(curImgsLDR);

curInLDR = curImgsLDR;

%%% concatenating inputs in the LDR and HDR domains
curInputs = curInLDR{1};
curInputs = cat(3, curInputs, LDRtoHDR(curInLDR{1}, curExpo(1)));
for k = 2 : numImgs
    curInputs = cat(3, curInputs, curInLDR{k});
    curInputs = cat(3, curInputs, LDRtoHDR(curInLDR{k}, curExpo(k)));
end
