clc
clear all
close all
% data acquisition of a robotic arm system
d = 0.01; % gridding parameter
weight_LW = 1; % weight on output matrix LW
% Robotic arm model
l1 = 10; % length of first arm
l2 = 7; % length of second arm
theta1_start = 0;
theta1_end = pi/2;
theta2_start = 0;
theta2_end = pi;
theta1 = theta1_start:d:pi/theta1_end; % all possible theta1 values
theta2 = theta2_start:d:theta2_end;
[THETA1,THETA2] = meshgrid(theta1,theta2);
X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2); % compute x coordinates
Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2); % training output data
inputData = [THETA1(:),THETA2(:)]';
outputData = [X(:),Y(:)]';
  for i= 1:size(outputData,2)
      for j=1:2
           outputset{1,i}(1,1) = outputData(1,i)-l1*d;
           outputset{1,i}(1,2) = outputData(1,i)+l1*d;
           outputset{1,i}(2,1) = outputData(2,i)-l2*d;
           outputset{1,i}(2,2)=  outputData(2,i)+l2*d;
     end
  end
rf = sqrt(2)*(l1+l2); % Lipschitz constant of robotic arm system
numNeuron = 20; % Size of nerual network ELM
% delta = 1;
% TF='ReLu';
% TF1='poslin';
delta =0.23;
TF='sig';
TF1='sig';

ELMNetwork = elmtrain_LipRidge(inputData,outputData, numNeuron,outputData,TF,0,d,delta); % train ELM using pinv
% Computer the maximum distance of samples 
X = inputData; % input of ELM
Y = elmpredict(X,ELMNetwork); % output of reduced size ELM
% Original LW is stored in ELMNetwork.weight{3} 
ELMNetwork1 = ELMNetwork;
ELMNetwork1.weight{2} = ELMNetwork.weight{3};
Y1= elmpredict(X,ELMNetwork1);% output of orginal ELM

%% Multilayer BP neural network
% BP=newff(input,output,[20 20],{'poslin','purelin'});
% BP.trainParam.epochs = 1000;
% BP.trainParam.goal = 1e-5;
% BP.trainParam.lr = 0.01;
% BP= train(BPs,inputData,outputData);
% BPrigde = BPLipRdge(BP);

%% Training
d2 = sqrt(2)*d; % ||x_i-x_{i-1}||_2
%% e_max_lip for orginal ELM
% Lipschitz method
tic
dist_sample = max(vecnorm(Y1-outputData));
W1 = ELMNetwork.weight{1};
W2 = ELMNetwork.weight{2};
[n1,~] = size(W1);
[n2,~] = size(W2);
r = delta*norm(W1,2)*norm(W2,2);% trival method 
%r = lip_ne(ELMNetwork) % LipSDP method
e_max_Lip = dist_sample+rf*d2+r*d2; % max distance of all outputs, Lipschtiz method
toc
%% e_max_lip for reduced size ELM
% Lipschitz method
tic
dist_sample = max(vecnorm(Y-outputData));
W1 = ELMNetwork.weight{1};
W2 = ELMNetwork.weight{3};
[n1,~] = size(W1);
[n2,~] = size(W2);
r =delta*norm(W1,2)*norm(W2,2);% trival method 
%r = lip_ne(ELMNetwork) % LipSDP method
e_max_Lip1 = dist_sample+rf*d2+r*d2; % max distance of all outputs, Lipschtiz method
toc
%% Compute output set for each box for orginal ELM
W{1} =  ELMNetwork.weight{1};  
W{2} =  ELMNetwork.weight{3};  
b{1} =  ELMNetwork.bias{1};
b{2} =  ELMNetwork.bias{2}; 
activeFcn =  {TF1,'purelin'};
ffnn = ffnetwork(W,b,activeFcn);
options.tol = d;
inputIntvl = [theta1_start,theta1_end;theta1_start,theta2_end];
yInterval = outputSet(ffnn,inputIntvl,options);
lossmax=zeros(length(yInterval),size(ELMNetwork.bias{2},2));
distance=zeros(length(yInterval),1);
for j = 1:length(yInterval)
    for i= 1:size(ELMNetwork.bias{2},2)
        lossmax(j,i) = max([outputset{1,j}(i,2)-yInterval{1,j}(i,1),yInterval{1,j}(i,2)-outputset{1,j}(i,1)]);
    end
end
for i=1:length(yInterval)
   distanceELM(i,1)=norm(lossmax(i,:)',2);
end
%Find min&max
e_max_reach1 = max(distanceELM);
%% Compute output set for each box Reduce-sized ELM
W{1} =  ELMNetwork.weight{1};  
W{2} =  ELMNetwork.weight{2};  
b{1} =  ELMNetwork.bias{1};
b{2} =  ELMNetwork.bias{2}; 
activeFcn =   {TF1,'purelin'};
ffnn = ffnetwork(W,b,activeFcn);
options.tol = d;
inputIntvl = [theta1_start,theta1_end;theta1_start,theta2_end];
tic
yInterval = outputSet(ffnn,inputIntvl,options);
lossmax=zeros(length(yInterval),size(ELMNetwork.bias{2},2));
distance=zeros(length(yInterval),1);
for j = 1:length(yInterval)
    for i= 1:size(ELMNetwork.bias{2},2)
        lossmax(j,i) = max([outputset{1,j}(i,2)-yInterval{1,j}(i,1),yInterval{1,j}(i,2)-outputset{1,j}(i,1)]);
    end
end
for i=1:length(yInterval)
   distanceRid(i,1)=norm(lossmax(i,:)',2);
end
%Find min&max
toc
e_max_reach = max(distanceRid);%% Compute output set for each box Reduce-sized ELM
%% E_max_reachlip of reduced sized ELM
% Error=zeros(size(Y,1),size(Y,2));
% tic
% for i=1:size(Y,1)
%     for j = 1:size(Y,2)
%         Error(i,j)=max([Y(i,j)-outputset{1,j}(i,1),outputset{1,j}(i,2)-Y(i,j)]);
%     end
% end
% loss = max(vecnorm(Error))+d*sqrt(2)*0.25*norm(ELMNetwork.weight{2},2)*norm(ELMNetwork.weight{1},2);
% e_max_reachlip=loss;
% toc
 % max distance of all outputs, reach set method
%% E_max_reachlip1 of orginal ELM
%  Error=zeros(size(Y1,1),size(Y1,2));
% for i=1:size(Y1,1)
%     for j = 1:size(Y1,2)
%         Error(i,j)=max([Y1(i,j)-outputset{1,j}(i,1),outputset{1,j}(i,2)-Y1(i,j)]);
%     end
% end
% loss = max(vecnorm(Error))+d*sqrt(2)*0.25*norm(ELMNetwork.weight{3},2)*norm(ELMNetwork.weight{1},2);
% e_max_reachlip1=loss;
%% Figures
%% a.Plot reach method results using SPSA
numPoints = 500;
theta1_rand = (theta1_end-theta1_start).*rand(1,numPoints) + theta1_start;
theta2_rand = (theta2_end-theta2_start).*rand(1,numPoints) + theta2_start;
input_rand = [theta1_rand;theta2_rand]; % random numPoints inputs
output_elm = elmpredict(input_rand,ELMNetwork); % random numPoints outputs
X_f = l1 * cos(theta1_rand) + l2 * cos(theta1_rand + theta2_rand); 
Y_f = l1 * sin(theta1_rand) + l2 * sin(theta1_rand + theta2_rand); % model output data
figure('NumberTitle', 'off', 'Name', 'Set-valued Reachability: Optimized ELM')
plot(output_elm(1,:),output_elm(2,:),'*')
hold on
plot(output_elm(1,:),output_elm(2,:),'o')
hold on
for i = 1:1:numPoints
    center = output_elm(:,i)';
    circle(center,e_max_reach,1000,'--');
    hold on
end
plot(output_elm(1,:),output_elm(2,:),'o')
plot(X_f,Y_f,'*')
legend('Actual Position','Predict Position','Error bound')
axis equal

%legend('Estimated Bound','ELM','Robotic Arm')
%legend('Location','southoutside')
%title(['Set-valued Reachability: ELM using SPSA Trained with ',num2str(numNeuron),' Neurons, Distance <' num2str(e_max_reach)])

%% b.Plot reach method results(original ELM)

output_elm = elmpredict(input_rand,ELMNetwork1); % random numPoints outputs
X_f = l1 * cos(theta1_rand) + l2 * cos(theta1_rand + theta2_rand); 
Y_f = l1 * sin(theta1_rand) + l2 * sin(theta1_rand + theta2_rand); % model output data
figure('NumberTitle', 'off', 'Name', 'Set-valued Reachability: ELM')
plot(output_elm(1,:),output_elm(2,:),'*')
hold on
plot(output_elm(1,:),output_elm(2,:),'o')
hold on

for i = 1:1:numPoints
    center = output_elm(:,i)';
    circle(center,1.9494,1000,'--');
    hold on
end
plot(output_elm(1,:),output_elm(2,:),'o')
plot(X_f,Y_f,'*')
legend('真实位置','预测位置','保证误差')
legend('FontSize', 12)
axis equal
%legend('Estimated Bound','ELM','Robotic Arm')
%legend('Location','southoutside')
%title(['Set-valued Reachability: ELM  Trained with ',num2str(numNeuron),' Neurons, Distance <' num2str(e_max_reach1)])

%% c.Plot Lipschitz method results (reduced size ELM)
output_elm = elmpredict(input_rand,ELMNetwork); % random numPoints outputs
X_f = l1 * cos(theta1_rand) + l2 * cos(theta1_rand + theta2_rand); 
Y_f = l1 * sin(theta1_rand) + l2 * sin(theta1_rand + theta2_rand); % model output data
figure('NumberTitle', 'off', 'Name', 'Lipschitz Method: Optimized ELM')
plot(output_elm(1,:),output_elm(2,:),'*')
hold on
plot(output_elm(1,:),output_elm(2,:),'o')
hold on
for i = 1:1:numPoints
    center = output_elm(:,i)';
    circle(center,e_max_Lip,1000,'--');
    hold on
end
plot(X_f,Y_f,'*')
plot(output_elm(1,:),output_elm(2,:),'o')
legend('Actual Position','Predict Position','Error bound')
axis equal
%title(['Lipschitz Method: ELM using SPSA Trained with ',num2str(numNeuron),' Neurons, Distance <' num2str(e_max_Lip)])
%% d.Plot Lipschitz method results
output_elm = elmpredict(input_rand,ELMNetwork1); % random numPoints outputs
X_f = l1 * cos(theta1_rand) + l2 * cos(theta1_rand + theta2_rand); 
Y_f = l1 * sin(theta1_rand) + l2 * sin(theta1_rand + theta2_rand); % model output data
figure('NumberTitle', 'off', 'Name', 'Lipschitz Method: ELM')
plot(output_elm(1,:),output_elm(2,:),'*')
hold on
plot(output_elm(1,:),output_elm(2,:),'o')
hold on
for i = 1:1:numPoints
    center = output_elm(:,i)';
    circle(center,e_max_Lip1,1000,'--');
    hold on
end
plot(X_f,Y_f,'*')
plot(output_elm(1,:),output_elm(2,:),'o')
legend('Actual Position','Predict Position','Error bound')
axis equal
%title(['Lipschitz Method: ELM Trained with ',num2str(numNeuron),' Neurons, Distance <' num2str(e_max_Lip1)])
%% e.Plot Set-valued reachability (SPSA)
% output_elm = elmpredict(input_rand,ELMNetwork); % random numPoints outputs
% X_f = l1 * cos(theta1_rand) + l2 * cos(theta1_rand + theta2_rand); 
% Y_f = l1 * sin(theta1_rand) + l2 * sin(theta1_rand + theta2_rand); % model output data
% figure('NumberTitle', 'off', 'Name','Set-valued Reachability & Lipschitz Method: ELM using SPSA')
% % plot(output_elm(1,:),output_elm(2,:),'*')
% plot(X_f,Y_f,'*')
% hold on
% plot(output_elm(1,:),output_elm(2,:),'o')
% hold on
% for i = 1:1:numPoints
%     center = output_elm(:,i)';
%     circle(center,e_max_reachlip,1000,'--');
%     hold on
% end
% plot(X_f,Y_f,'*')
% plot(output_elm(1,:),output_elm(2,:),'o')
% legend('Actual Position','Predict Position','Error bound')
% axis equal
%title(['Set-valued Reachability & Lipschitz Method: ELM using SPSA Trained with ',num2str(numNeuron),' Neurons, Distance <' num2str(e_max_reachlip)])
%% f.Plot Set-valued reachability
% output_elm = elmpredict(input_rand,ELMNetwork1); % random numPoints outputs
% X_f = l1 * cos(theta1_rand) + l2 * cos(theta1_rand + theta2_rand); 
% Y_f = l1 * sin(theta1_rand) + l2 * sin(theta1_rand + theta2_rand); % model output data
% figure('NumberTitle', 'off', 'Name','Set-valued Reachability & Lipschitz Method: ELM')
% plot(X_f,Y_f,'*')
% hold on
% plot(output_elm(1,:),output_elm(2,:),'o')
% hold on
% for i = 1:1:numPoints
%     center = output_elm(:,i)';
%     circle(center,e_max_reachlip1,1000,'--');
%     hold on
% end
% plot(X_f,Y_f,'*')
% plot(output_elm(1,:),output_elm(2,:),'o')
% legend('Actual Position','Predict Position','Error bound')
% axis equal
%title(['Set-valued Reachability & Lipschitz Method: ELM Trained with ',num2str(numNeuron),' Neurons, Distance <' num2str(e_max_reachlip1)])

fprintf(['The Error of ||LW*H-T|| = ',  num2str(dist_sample),'.\n'])
fprintf(['Distance (Reachable set method) between ELM and original model is dist_reach = ',  num2str(e_max_reach),'.\n'])
fprintf(['Distance (Lipschitz method) between ELM and original model is dist_Lip = ',  num2str(e_max_Lip),'.\n'])
%% mse
%1. Reduced=sized ELM
z=Y-outputData;
k=0;
for i = 1:size(Y,2)
k1=z(1,i)^2+z(2,i)^2;
    k=k+k1;
end
mse_ELMLipRidge=k/size(Y,2);
%2. ELM
z=Y1-outputData;
k=0;
for i = 1:size(Y,2)
k1=z(1,i)^2+z(2,i)^2;
    k=k+k1;
end
mse_ELM=k/size(Y,2);
%e_max_reach
%e_max_Lip
%dist_sample
