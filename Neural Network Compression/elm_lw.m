function ELMNetwork1=elm_lw(inputData,outputData,ELMNetwork)
%% Hidden Layers
% Randomly Generate the Input Weight Matrix
ELMNetwork1 = ELMNetwork;
Bias = repmat(ELMNetwork.bias{1},1,size(inputData,2));
tempH = ELMNetwork.weight{1}* inputData+Bias;
H = 1 ./ (1 + exp(-tempH));
ELMNetwork1.weight{2} = (pinv(H') * outputData')';
end