clc
clear
close all
d = 1e-3;
val=200;
weight_LW = 0; % weight on output matrix LW
numNeuron = 50; % Size of nerual network ELM
inputIntvl = [-10,10];
numInput = 1;
numOutput = 1;
numNN = [100,100,100,numOutput];
% delta = 1;
% TF='ReLu';
% TF1='poslin';
delta =0.23;
TF='sig';
TF1='sig';
activeFcn = {TF1,TF1,TF1,'purelin'};
W{1} = randn(numNN(1),numInput);
b{1} = randn(numNN(1),1);
for i = 2:1:length(numNN)
    W{i} = randn(numNN(i),numNN(i-1));
     b{i} = randn(numNN(i),1);
end
ffnn = ffnetwork(W,b,activeFcn); % orginal neural network 
r=1;
% Lipschtiz constant for ffnn
for i = 1:1:length(W)-1
    [n,~] = size(W{i});
    r = delta*norm(W{i},2)*norm(W{i+1},2)*r;
end
[n,~] = size(W{end});

% Reachable Set of Original NN
options.tol = d;
yInterval_ffnn = outputSet(ffnn,inputIntvl,options);

gamma_f=r;
inputData = inputIntvl(1):d:inputIntvl(2);
outputData = outputSingle(ffnn,inputData);
%plot(inputData,outputData,'*')
%hold on

% ELMNetwork = elmtrain_msal(inputData,outputData, numNeuron, weight_LW,'sig',0,tol); % train ELM
ELMNetwork = elmtrain_Lipridge(inputData,outputData, numNeuron,TF,0,d,delta); % train ELM

X = inputData; % input of ELM
Y = elmpredict(X,ELMNetwork); % output of ELM
%plot(X,Y,'*')
dist_sample = max(abs(Y-outputData));
%% Original ELM
ELMNetwork1 = elm_lw(inputData,outputData,ELMNetwork);
% W1 = ELMNetwork.weight{1};
% W2 = ELMNetwork.weight{2};
% [n1,~] = size(W1);
% [n2,~] = size(W2);
% r_elm = sqrt(n1)*norm(W1,2)*sqrt(n2)*norm(W2,2);% trivial method for Lipschitiz of ELM
r_elm = delta*norm(ELMNetwork.weight{1},2)*norm(ELMNetwork.weight{2},2);
r_elm1= delta*norm(ELMNetwork1.weight{1},2)*norm(ELMNetwork1.weight{2},2);
% Reachable Set of ELM

%% Compute output set for each box
W_elm{1} =  ELMNetwork.weight{1};  
W_elm{2} =  ELMNetwork.weight{2};  
b_elm{1} =  ELMNetwork.bias{1};
b_elm{2} =  ELMNetwork.bias{2}; 
activeFcn = {TF1,'purelin'};
ffnn_elm = ffnetwork(W_elm,b_elm,activeFcn);
options.tol = d;
yInterval_elm = outputSet(ffnn_elm,inputIntvl,options);
for i = 1:1:length(yInterval_elm)   
    y_dist_elm(i) = max(abs([yInterval_ffnn{i}(:,2) - yInterval_elm{i}(:,1),yInterval_elm{i}(:,2)-yInterval_ffnn{i}(:,1)])); % length of each output box
end
y_dist_max_elm = max(y_dist_elm); 

e_max_reach = y_dist_max_elm; % max distance of all outputs, reach set method
e_max_Lip = dist_sample+r*d+r_elm*d; % max distance of all outputs, Lipschtiz method

%%
% Compute output set for each box
W_elm1{1} =  ELMNetwork1.weight{1};  
W_elm1{2} =  ELMNetwork1.weight{2};  
b_elm1{1} =  ELMNetwork1.bias{1};
b_elm1{2} =  ELMNetwork1.bias{2}; 
ffnn_elm1 = ffnetwork(W_elm1,b_elm1,activeFcn);
options.tol = d;
yInterval_elm = outputSet(ffnn_elm1,inputIntvl,options);
for i = 1:1:length(yInterval_elm)
    y_dist_elm1(i) =max(abs([yInterval_ffnn{i}(:,2) - yInterval_elm{i}(:,1),yInterval_elm{i}(:,2)-yInterval_ffnn{i}(:,1)]));; % length of each output box
end
y_dist_max_elm1 = max(y_dist_elm1); 

e_max_reach1 = y_dist_max_elm1; % max distance of all outputs, reach set method
e_max_Lip1 = dist_sample+r*d+r_elm1*d; % max distance of all outputs, Lipschtiz method
%%
a = inputIntvl(1);
b = inputIntvl(2);
numPoints = 10000;
X = (b-a).*rand(1,numPoints) + a;
Y = outputSingle(ffnn,X);
Y_elm = elmpredict(X,ELMNetwork);
Y_elm1 = elmpredict(X,ELMNetwork1);
%% plot figure
%1.
figure('NumberTitle', 'off', 'Name', 'Reachable set method of Reduced-size ELM')
hold on
plot(X,Y,'.') % plot original NN
plot(X,Y_elm,'*') % plot ELM
Y_upper = Y_elm+e_max_reach; % compute upper bound (output of ELM plus distance), you can also use Y_upper = Y+e_max;
Y_lower = Y_elm-e_max_reach; % compute lower bound (output of minus distance), you can also use Y_lower = Y+e_max;
plot(X,Y_upper,'.') % plot estimated upper bound
plot(X,Y_lower,'.') % plot estimated lower bound
legend('Original NN','Reduced-size ELM','Estimated Upper Bound', 'Estimated Lower Bound')
legend('Location','southoutside')
%title({['Reachable Set Method'], ['Orignal NN with [', num2str(numNN), '] Neurons'], ['Reduced ELM with ',num2str(numNeuron),' Neurons, Distance < ' num2str(e_max_reach)]})
fprintf(['Distance (Reachable set method) between ELM and original NN is dist_reach = ',  num2str(e_max_reach),'.\n'])
%2.
figure('NumberTitle', 'off', 'Name', 'Lipschitz Method method of Reduced-size ELM')
hold on
plot(X,Y,'.') % plot original NN
plot(X,Y_elm,'*') % plot ELM
e_max_Lip=0.4788;
Y_upper = Y_elm+e_max_Lip; % compute upper bound (output of ELM plus distance), you can also use Y_upper = Y+e_max;
Y_lower = Y_elm-e_max_Lip; % compute lower bound (output of minus distance), you can also use Y_lower = Y+e_max;
plot(X,Y_upper,'.') % plot estimated upper bound
plot(X,Y_lower,'.') % plot estimated lower bound
legend('原始神经网络输出','压缩神经网络输出','保证误差上界', '保证误差下界')
legend('FontSize', 11)
%ylim([-350 350])
xlabel('模型输入','FontSize', 14)
ylabel('模型输出','FontSize', 14)
%legend('Location','southoutside')
%title({['Lipschitz Method'], ['Orignal NN with [', num2str(numNN), '] Neurons'], ['Reduced ELM with ',num2str(numNeuron),' Neurons, Distance < ' num2str(e_max_Lip)]})
%fprintf(['Distance (Lipschitz method) between ELM and original NN is dist_reach = ',  num2str(e_max_Lip),'.\n'])
%3.
figure('NumberTitle', 'off', 'Name', 'Oiginal ELM & Reduced-size ELM')
hold on
plot(X,Y,'.') % plot original NN
plot(X,Y_elm,'*') % plot ELM
plot(X,Y_elm1,'o');% plot Original ELM
legend('Original NN','Reduced-size ELM','Original ELM')
%4.
figure('NumberTitle', 'off', 'Name', 'Lipshcitz estimation of Oiginal ELM')
hold on
plot(X,Y,'.') % plot original NN
plot(X,Y_elm1,'*') % plot ELM
Y_upper = Y_elm1+e_max_Lip1; % compute upper bound (output of ELM plus distance), you can also use Y_upper = Y+e_max;
Y_lower = Y_elm1-e_max_Lip1; % compute lower bound (output of minus distance), you can also use Y_lower = Y+e_max;
plot(X,Y_upper,'.') % plot estimated upper bound
plot(X,Y_lower,'.') % plot estimated lower bound
legend('Original NN','Original ELM','Estimated Upper Bound', 'Estimated Lower Bound')
legend('Location','southoutside')
title({['Lipschitz Method'], ['Orignal NN with [', num2str(numNN), '] Neurons'], ['Reduced ELM with ',num2str(numNeuron),' Neurons, Distance < ' num2str(e_max_Lip1)]})
fprintf(['Distance (Lipschitz method) between ELM and original NN is dist_reach = ',  num2str(e_max_Lip1),'.\n'])
%5.
figure('NumberTitle', 'off', 'Name', 'reachable set of Original ELM')
hold on
plot(X,Y,'.') % plot original NN
plot(X,Y_elm1,'*') % plot ELM
Y_upper = Y_elm1+e_max_reach1; % compute upper bound (output of ELM plus distance), you can also use Y_upper = Y+e_max;
Y_lower = Y_elm1-e_max_reach1; % compute lower bound (output of minus distance), you can also use Y_lower = Y+e_max;
plot(X,Y_upper,'.') % plot estimated upper bound
plot(X,Y_lower,'.') % plot estimated lower bound
%legend('Original NN','Original ELM','Estimated Upper Bound', 'Estimated Lower Bound')
legend('原始神经网络输出','压缩神经网络输出','保证误差上界', '保证误差下界')
legend('FontSize', 14)
legend('Location','southoutside')
%title({['Reachable Set Method'], ['Orignal NN with [', num2str(numNN), '] Neurons'], ['Reduced ELM with ',num2str(numNeuron),' Neurons, Distance < ' num2str(e_max_reach1)]})
fprintf(['Distance (Reachable set method) between ELM and original NN is dist_reach = ',  num2str(e_max_reach1),'.\n'])








