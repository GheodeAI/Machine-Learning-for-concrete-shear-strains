clear all
close all
clc

load('predictores_pena.mat')
load('target_pena.mat')
% load('a1_ntrain.mat')
% load('a2_ntrain2.mat')
load('a1_train_aux.mat')
load('a2_train_aux.mat')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Limpiamos predictores y target
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
caract = predictores_pena;

x = isnan(target_pena(:,7)); % Valores de potencia
ind1 = find(x==1);
target_pena(ind1,:) = [];
caract(ind1,:) = [];
target = target_pena(:,5);  %Velocidad de viento


%% Levenberg_Marquardt
for n=1:10
    %Datos de entrenamiento y test
    a1 = a1_aux{1,n};
    n_train = n_train_aux{1,n};
    a2 = a2_aux{1,n};
    n_train2 = n_train2_aux{1,n};
    
    EntradaTrain = caract(a1(1:n_train),:);
    EntradaTest = caract(a1((n_train+1):end),:);
    TargetTrain = target(a1(1:n_train),:);
    TargetTest = target(a1((n_train+1):end),:);

    %Datos validación
    matrizEntradaTrain = EntradaTrain(a2(1:n_train2),:);
    matrizTargetTrain = TargetTrain(a2(1:n_train2),:);

    EntradaTrain(a2(1:n_train2),:) = [];
    TargetTrain(a2(1:n_train2),:) = [];

    net = feedforwardnet(2,'trainlm');
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 0;
    net.trainParam.max_fail = 6;
    net.trainParam.min_grad = 1e-7;
    net.trainParam.mu = 0.001;
    net.trainParam.mu_dec = 0.1;
    net.trainParam.mu_inc = 10;
    net.trainParam.mu_max = 1e10;
    net.trainParam.show = 25;
    net.trainParam.showCommandLine = false;
    net.trainParam.showWindow = true;
    net.trainParam.time = inf;

    [net,tr] = train(net,matrizEntradaTrain',matrizTargetTrain');
    % Salida train
    y_train = net(EntradaTrain');
    y_train_mlp = y_train';
    % Salida test
    y_test_mlp = net(EntradaTest');
    y_test_mlp = y_test_mlp';
    
    % Error
    traincc=sqrt(sum((TargetTrain-y_train_mlp).^2)/length(TargetTrain)); %%Error cuadratico medio de train
    
    % Salidas
    traincc_aux(n) = traincc;
    y_train_aux_mlp{:,n} = y_train_mlp;
    y_test_aux_mlp{:,n} = y_test_mlp;
    TargetTest_aux_mlp{:,n} = TargetTest;
    TargetTrain_aux_mlp{:,n} = TargetTrain;
end

[x pos] = min(traincc_aux);
traincc = x;

ypred_test_mlp = y_test_aux_mlp{pos};
y_train_mlp = y_train_aux_mlp{pos};
TargetTest = TargetTest_aux_mlp{pos};

xx = find(ypred_test_mlp<0);
ypred_test_mlp(xx) = [];
TargetTest(xx) = [];
yy = find(ypred_test_mlp>30);
ypred_test_mlp(yy) = [];
TargetTest(yy) = [];

save('ypred_test_mlp.mat','ypred_test_mlp');
save('y_train_mlp.mat','y_train_mlp');
save('target_train.mat','TargetTrain');
save('Conjunto_Test.mat','TargetTest','EntradaTest');
save('E:\Investigación_posdoctoral\ENSEMBLE_pena\Peñaparda\MLP\Resultados\DEFINITIVO\TargetTest_TargetTrain_ypredTrain_ypredTest_aux_mlp.mat','TargetTrain_aux_mlp','TargetTest_aux_mlp','y_train_aux_mlp','y_test_aux_mlp');

figure
plot(TargetTest)
hold on
plot(ypred_test_mlp, 'r')
title('MLP');

figure
plot(TargetTest,ypred_test_mlp,'.');
hold on;
plot(linspace(min([TargetTest;ypred_test_mlp]),max([TargetTest;ypred_test_mlp]),100),linspace(min([TargetTest;ypred_test_mlp]),max([TargetTest;ypred_test_mlp]),100),'r','linewidth',2);
title('MLP');
axis square

%% Metrica para el mejor caso (mínimo error)
Obs_MLP = TargetTest;   % obs values 
Pred_MLP = ypred_test_mlp;   % pred values 
RMSE_test = sqrt(sumsqr(Obs_MLP-Pred_MLP)/length(Obs_MLP));
RMSE_train = traincc_aux(pos);
%Correlacion
R2 = corr(Obs_MLP,Pred_MLP);

save('E:\Investigación_posdoctoral\ENSEMBLE_pena\Peñaparda\MLP\Resultados\DEFINITIVO\metrica.mat','RMSE_test','R2','RMSE_train');

%% Test no paramétrico
%Ranking error test
for in=1:length(y_test_mlp_aux)
    error_test_mlp(:,in) = sqrt(sum((TargetTest_aux{:,in}-y_test_mlp_aux{:,in}).^2)/length(TargetTest_aux{:,in}));
end
% error_test_ordenado = sort(error_test,'ascend');
% media_error_test = mean(error_test_ordenado);

%Media del ranking
%Ranking errer train
error_train_mlp = traincc_aux;
% error_train_ordenado = sort(error_train,'ascend');
% media_error_test = mean(error_train_ordenado);

save('E:\Investigación_posdoctoral\ENSEMBLE_pena\Peñaparda\MLP\Resultados\DEFINITIVO\errores_mlp.mat','error_test_mlp','error_train_mlp');





