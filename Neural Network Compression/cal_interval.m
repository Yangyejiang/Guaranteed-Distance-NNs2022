function Y=cal_interval(P,T,ELMNetwork,X)
%Caculate the interval of the outputset
%-Input £º
%
% P    - Input of the Training set
% T    - Outpt of the Training set
% ELM_network  - Struct of ELM network. Example: ELM_network={IW,Bias,LW,Activ}
%-Output :
% Y    - Output interval        % [ymin ymax] belongs to [xmin,xmax]
%-Dimentions dictionary:
%
% P(R*Q) T(S*Q)
% X(R,2) % Pertubation interval
% IW(N*R) LW(S*N) B(N*1)
%-Network: max.||(LW*activ(IW*(P+u)+remat(B))-T||
%-Example:
% Y=cal_interval(P,T,ELM_network,X)
if nargin < 4
    error('cal_interval','Not enough input arguments.');
end
%% Read the parameters
[R,~] = size(P);
[S,Q] = size(T);
N = size(ELMNetwork.weight{1},1);
%% Augmented Bias Matirx and perturbations Matrix
Bias = repmat(ELMNetwork.bias{1},1,Q);
w_l = X(:,1);
w_u = X(:,2); 
W_l = repmat(w_l,1,Q);
W_u = repmat(w_u,1,Q);
%% Caculate the maximum and minimum value of the obj.function
P_min = P+ W_l;
P_max = P+ W_u;
% Initialize Hidden Layer & Output Layer
H_l = zeros(N,Q);
H_u = zeros(N,Q);
T1_l = zeros(S,Q);
T1_u = zeros(S,Q);
% Hidden Layer input upper and lower bounds 
for i = 1:Q
    for j = 1:N
        for k = 1:R
            if ELMNetwork.weight{1}(j,k) > 0
              H_l(j,i)=(ELMNetwork.weight{1}(j,k)*(P_min(k,i)))+H_l(j,i);
              H_u(j,i)=(ELMNetwork.weight{1}(j,k)*(P_max(k,i)))+H_u(j,i);
            else
              H_l(j,i)=(ELMNetwork.weight{1}(j,k)*P_max(k,i))+H_l(j,i);
              H_u(j,i)=(ELMNetwork.weight{1}(j,k)*P_min(k,i))+H_u(j,i);  
            end
        end
    end
end
% Activiation function
H_l = 1 ./ (1 + exp(-(H_l+Bias)));
H_u = 1 ./ (1 + exp(-(H_u+Bias)));
% Output Layer upper and lower bounds 
for i = 1:Q
    for j = 1:S
        for k = 1:N
            if ELMNetwork.weight{2}(j,k) > 0
              T1_l(j,i) = (ELMNetwork.weight{2}(j,k)*H_l(k,i)) + T1_l(j,i);
              T1_u(j,i) = (ELMNetwork.weight{2}(j,k)*H_u(k,i)) + T1_u(j,i);
            else
              T1_l(j,i) = (ELMNetwork.weight{2}(j,k)*H_u(k,i)) + T1_l(j,i);
              T1_u(j,i) = (ELMNetwork.weight{2}(j,k)*H_l(k,i)) + T1_u(j,i);  
            end
        end
    end
end
%% Caculate obj.function
% Output interval
ymin = zeros(S,Q);
ymax = zeros(S,Q);
for i = 1:S
    for j = 1:Q
        if((T1_l(i,j)<T(i,j))&&(T(i,j)<T1_u(i,j)))
            ymin(i,j) = 0;
            ymax(i,j) = max([abs(T1_l(i,j)-T(i,j)),abs(T1_u(i,j)-T(i,j))]);
        end    
        if((T(i,j)<T1_l(i,j)))
            ymin(i,j) =T1_l(i,j)-T(i,j);
            ymax(i,j) =T1_u(i,j)-T(i,j);
        end
        if (T(i,j)>T1_u(i,j))
            ymin(i,j) =T(i,j)-T1_u(i,j);
            ymax(i,j) =T(i,j)-T1_l(i,j);
        end
    end            
end
%Find min&max
Ymin = min(min(ymin));
Ymax = max(max(ymax));
Y = [Ymin,Ymax];
end