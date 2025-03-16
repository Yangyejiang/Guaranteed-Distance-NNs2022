function ELMNetwork = elmtrain_LipRidge(P,T,N,NN_ouputset,TF,TYPE,d,delta)
% ELMTRAIN Create and Train a Extreme Learning Machine
% Syntax
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE£¬tol)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% T   - Output Matrix of Training Set (S*Q)
% N   - Number of Hidden Neurons (default = Q)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (S*N)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0,tol)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)

%% Check
if nargin < 2
    error('ELM:Arguments','Not enough input arguments.');
end
if nargin < 3
    N = size(P,2);
end
if nargin < 4
    TF = 'sig';
end
if nargin < 5
    TYPE = 0;
end
if size(P,2) ~= size(T,2)
    error('ELM:Arguments','The columns of P and T must be same.');
end
[R,Q] = size(P);
if TYPE  == 1
    T  = ind2vec(T);
end
[S,Q] = size(T);
%% Hidden Layers
% Randomly Generate the Input Weight Matrix
IW = randn(N,R);                                         % % Randomly Generate the Bias Matrix
B = rand(N,1);
Bias = repmat(B,1,Q);
%% Calculate the Layer Output Matrix H
tempH = IW * P+Bias;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
    case 'ReLu'
        H = max(tempH,0);
end
%% Initial value of LW
LW = pinv(H') * T';
ELMNetwork.weight{1} = IW;  %Input weight
ELMNetwork.weight{2} = LW';  %Output weight
ELMNetwork.weight{3} = LW';
[~,S2] = size(LW);
ELMNetwork.bias{1} = B;
ELMNetwork.bias{2} = zeros(S2,1);
ELMNetwork.activeFcn = {TF  'purelin'};
ELMNetwork.layerNum = 2;
% Y = elmpredict(P,ELMNetwork);
% figure('NumberTitle', 'off', 'Name', 'Initial Trained Erros')
% plot(P,Y-T);
% xlabel('P');
% ylabel('Initial Trained Erros');
%% Initial Network
% ELMNetwork1=ELMNetwork;
%% SPSA Algorithm to evalue the LW s.t min dist.||nn-ELMNetwork||
% LW=zeros(N,S);
% for n = 1:size(ELMNetwork.weight{2},1)
% %tic;
% %Lipshcitz & Ridge Regression 
%     %LW=SPSA(P,NN_ouputset,ELMNetwork,val,n,d);
     lambda = delta*norm(IW)*1*sqrt(2)*d;
%     LW(:,n) = pinv(H*H'+lambda)*(H*T(n,:)');
% % Reformulate Network
% end
fprintf('the value of Ridge Regression k')
disp(lambda)

%toc
%% Yalmip solver(mosek)
for i = 1:S
LWso=sdpvar(N,1);
Obj=norm(H'*LWso-T(i,:)',2)+lambda*norm(LWso,2);
optimize([],Obj);
LW(:,i)=value(LWso);
end
fprintf('the value of Guaranteed Error of Lipschitz')
disp(norm(H'*LW-T',inf)+lambda*norm(LW,inf))
ELMNetwork.weight{2} = LW';
end



