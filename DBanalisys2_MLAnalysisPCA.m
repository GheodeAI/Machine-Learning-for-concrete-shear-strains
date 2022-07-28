%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Data analysis: Train & Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear all;close all;

% Read Data
addpath('MLP'); addpath('utils'); addpath('data');
dt = xlsread('ANN_fortraining_cro.xlsx');

%%% PCA
PCA=dt(:,1:end-2);
nKs = 10; nKmin = 1; nKmax = 25; R2_pca_t = [];
nT = size(PCA,1); nK=4;
[coeff,score,latent,tsquared,explained,mu] = pca(PCA);
lmdX = (PCA-mu)*coeff; nT = size(PCA,1);
Xm = PCA-mean(PCA); [U,S,V] = svd(Xm);
s=diag(S); sK = diag([s(1:nKs)' zeros(1,nT-nKs)]); SK = S; SK(1:size(S,1),1:size(S,1))=sK;
Vpca = V(:,1:nK);
% 1) 
lmdX2 = Xm*Vpca;

rng(1);
prTst = 0.2; [m,n]=size(dt);
idxTst = randperm(round(prTst*m));
idxTrn = setdiff(1:m,idxTst);
Xtrn  = lmdX2(idxTrn,:); Y1trn  = dt(idxTrn,end-1); Y2trn  = dt(idxTrn,end);
Xtst  = lmdX2(idxTst,:); Y1tst  = dt(idxTst,end-1); Y2tst  = dt(idxTst,end);
%%%


% %% Train and Test RFs
% % 1
% Ytrn = Y1trn; Ytst = Y1tst;
% Mdl = TreeBagger(100,Xtrn,Ytrn,'Method','regression');
% ypred_rf_tst = predict(Mdl,Xtst);
% 
% figure; stem(ypred_rf_tst,'b');
% 
% R2_t.corr_rf_tst = corr(ypred_rf_tst,Ytst)
% R2_t.rmse_rf_tst = sqrt(mean((ypred_rf_tst-Ytst).^2))
% R2_t.mae_rf_tst  = mean(abs(ypred_rf_tst-Ytst))
% 
% hbxplt1 = figure; boxplot(ypred_rf_tst-Ytst);
% 
% Y1prd_rf = ypred_rf_tst;
% 
% % 2
% Ytrn = Y2trn; Ytst = Y2tst;
% Mdl = TreeBagger(100,Xtrn,Ytrn,'Method','regression');
% ypred_rf_tst = predict(Mdl,Xtst);
% 
% figure; stem(ypred_rf_tst,'b');
% 
% R2_t.corr_rf_tst = corr(ypred_rf_tst,Ytst)
% R2_t.rmse_rf_tst = sqrt(mean((ypred_rf_tst-Ytst).^2))
% R2_t.mae_rf_tst  = mean(abs(ypred_rf_tst-Ytst))
% 
% hbxplt2 = figure; boxplot(ypred_rf_tst-Ytst);
% 
% Y2prd_rf = ypred_rf_tst;
% 
% Yprd_rf = [Y1prd_rf,Y2prd_rf];
% 
% % writematrix(Yprd_rf,'RF_pred.xls');
%% Train and Test Linear Regressors
% 1
Ytrn = Y1trn; Ytst = Y1tst;
B_r = regress(Ytrn,Xtrn);
[B_lss,FitInfo] = lasso(Xtrn,Ytrn,'CV',10);
    figure;
    lassoPlot(B_lss,FitInfo,'PlotType','CV');
    legend('show') % Show legend
B_lss=B_lss(:,FitInfo.IndexMinMSE);
% B_rdg = ridge(Xtrn,Ytrn);

ypred_lr_tst = Xtst*B_r;
ypred_ls_tst = Xtst*B_lss;

R2_t.corr_lr_tst = corr(ypred_lr_tst,Ytst)
R2_t.rmse_lr_tst = sqrt(mean((ypred_lr_tst-Ytst).^2))
R2_t.mae_lr_tst  = mean(abs(ypred_lr_tst-Ytst))

R2_t.corr_ls_tst = corr(ypred_ls_tst,Ytst)
R2_t.rmse_ls_tst = sqrt(mean((ypred_ls_tst-Ytst).^2))
R2_t.mae_ls_tst  = mean(abs(ypred_ls_tst-Ytst))

% figure(hbxplt1); hold on; boxplot(ypred_svm_lr_tst-Ytst);
% figure(hbxplt1); hold on; boxplot(ypred_svm_ls_tst-Ytst);

figure; hold on
stem(ypred_lr_tst,'b');
stem(ypred_ls_tst,'g');
% stem(ypred_svm_rg_tst,'m');

Y1prd_lr = ypred_lr_tst;
Y1prd_ls = ypred_ls_tst;
% Y1prd_rg = ypred_svm_rg_tst;

% 2
Ytrn = Y2trn; Ytst = Y2tst;
B_r = regress(Ytrn,Xtrn);
[B_lss,FitInfo] = lasso(Xtrn,Ytrn,'CV',10);
    figure;
    lassoPlot(B_lss,FitInfo,'PlotType','CV');
    legend('show') % Show legend
B_lss=B_lss(:,FitInfo.IndexMinMSE);
% B_rdg = ridge(Xtrn,Ytrn);

ypred_lr_tst = Xtst*B_r;
ypred_ls_tst = Xtst*B_lss;

R2_t.corr_lr_tst = corr(ypred_lr_tst,Ytst)
R2_t.rmse_lr_tst = sqrt(mean((ypred_lr_tst-Ytst).^2))
R2_t.mae_lr_tst  = mean(abs(ypred_lr_tst-Ytst))

R2_t.corr_ls_tst = corr(ypred_ls_tst,Ytst)
R2_t.rmse_ls_tst = sqrt(mean((ypred_ls_tst-Ytst).^2))
R2_t.mae_ls_tst  = mean(abs(ypred_ls_tst-Ytst))

% figure(hbxplt2); hold on; boxplot(ypred_svm_lr_tst-Ytst);
% figure(hbxplt2); hold on; boxplot(ypred_svm_ls_tst-Ytst);

figure; hold on
stem(ypred_lr_tst,'b');
stem(ypred_ls_tst,'g');
% stem(ypred_svm_rg_tst,'m');

Y2prd_lr = ypred_lr_tst;
Y2prd_ls = ypred_ls_tst;
% Y2prd_rg = ypred_svm_rg_tst;

Yprd_lr = [Y1prd_lr,Y2prd_lr];
Yprd_ls = [Y1prd_ls,Y2prd_ls];

writematrix(Yprd_lr,'LREG_pred.xls');
writematrix(Yprd_ls,'LASSO_pred.xls');

return;
% %% Train and Test SVRs
% % 1
% Ytrn = Y1trn; Ytst = Y1tst;
% 
% SVMModel_g = fitrsvm(Xtrn,Ytrn,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
% SVMModel_l = fitrsvm(Xtrn,Ytrn,'KernelFunction','linear','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
% SVMModel_p = fitrsvm(Xtrn,Ytrn,'KernelFunction','polynomial','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
% 
% % test
% ypred_svm_g_tst = predict(SVMModel_g,Xtst);
% ypred_svm_l_tst = predict(SVMModel_l,Xtst);
% ypred_svm_p_tst = predict(SVMModel_p,Xtst);
% 
% R2_t.corr_svm_g_tst = corr(ypred_svm_g_tst,Ytst)
% R2_t.rmse_svm_g_tst = sqrt(mean((ypred_svm_g_tst-Ytst).^2))
% R2_t.mae_svm_g_tst  = mean(abs(ypred_svm_g_tst-Ytst))
% 
% R2_t.corr_svm_l_tst = corr(ypred_svm_l_tst,Ytst)
% R2_t.rmse_svm_l_tst = sqrt(mean((ypred_svm_l_tst-Ytst).^2))
% R2_t.mae_svm_l_tst  = mean(abs(ypred_svm_l_tst-Ytst))
% 
% R2_t.corr_svm_p_tst = corr(ypred_svm_p_tst,Ytst)
% R2_t.rmse_svm_p_tst = sqrt(mean((ypred_svm_p_tst-Ytst).^2))
% R2_t.mae_svm_p_tst  = mean(abs(ypred_svm_p_tst-Ytst))
% 
% % figure(hbxplt1); hold on; boxplot(ypred_svm_g_tst-Ytst);
% % figure(hbxplt1); hold on; boxplot(ypred_svm_l_tst-Ytst);
% % figure(hbxplt1); hold on; boxplot(ypred_svm_p_tst-Ytst);
% 
% 
% figure; hold on
% stem(ypred_svm_g_tst,'b');
% stem(ypred_svm_l_tst,'g');
% stem(ypred_svm_p_tst,'m');
% 
% Y1prd_g = ypred_svm_g_tst;
% Y1prd_l = ypred_svm_l_tst;
% Y1prd_p = ypred_svm_p_tst;
% 
% % 2
% Ytrn = Y2trn; Ytst = Y2tst;
% 
% SVMModel_g = fitrsvm(Xtrn,Ytrn,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
% SVMModel_l = fitrsvm(Xtrn,Ytrn,'KernelFunction','linear','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
% SVMModel_p = fitrsvm(Xtrn,Ytrn,'KernelFunction','polynomial','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
% 
% % test
% ypred_svm_g_tst = predict(SVMModel_g,Xtst);
% ypred_svm_l_tst = predict(SVMModel_l,Xtst);
% ypred_svm_p_tst = predict(SVMModel_p,Xtst);
% 
% R2_t.corr_svm_g_tst = corr(ypred_svm_g_tst,Ytst)
% R2_t.rmse_svm_g_tst = sqrt(mean((ypred_svm_g_tst-Ytst).^2))
% R2_t.mae_svm_g_tst  = mean(abs(ypred_svm_g_tst-Ytst))
% 
% R2_t.corr_svm_l_tst = corr(ypred_svm_l_tst,Ytst)
% R2_t.rmse_svm_l_tst = sqrt(mean((ypred_svm_l_tst-Ytst).^2))
% R2_t.mae_svm_l_tst  = mean(abs(ypred_svm_l_tst-Ytst))
% 
% R2_t.corr_svm_p_tst = corr(ypred_svm_p_tst,Ytst)
% R2_t.rmse_svm_p_tst = sqrt(mean((ypred_svm_p_tst-Ytst).^2))
% R2_t.mae_svm_p_tst  = mean(abs(ypred_svm_p_tst-Ytst))
% 
% % figure(hbxplt2); hold on; boxplot(ypred_svm_g_tst-Ytst);
% % figure(hbxplt2); hold on; boxplot(ypred_svm_l_tst-Ytst);
% % figure(hbxplt2); hold on; boxplot(ypred_svm_p_tst-Ytst);
% 
% figure; hold on
% stem(ypred_svm_g_tst,'b');
% stem(ypred_svm_l_tst,'g');
% stem(ypred_svm_p_tst,'m');
% 
% Y2prd_g = ypred_svm_g_tst;
% Y2prd_l = ypred_svm_l_tst;
% Y2prd_p = ypred_svm_p_tst;
% 
% Yprd_g = [Y1prd_g,Y2prd_g];
% Yprd_l = [Y1prd_l,Y2prd_l];
% Yprd_p = [Y1prd_p,Y2prd_p];
% 
% % writematrix(Yprd_g,'SVR_g_pred.xls');
% % writematrix(Yprd_l,'SVR_l_pred.xls');
% % writematrix(Yprd_p,'SVR_p_pred.xls');
% 
% return
%% Train and Test MLPs
% 1
Ytrn = Y1trn; Ytst = Y1tst;
mlpNit=25; mlpD=zeros(length(Ytrn),mlpNit);
net = feedforwardnet([4],'trainlm');
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
net.trainParam.showWindow = false;
net.trainParam.time = inf;
for l=1:50%mlpNit
    [net,tr] = train(net,Xtrn',Ytrn');
    mlpD(:,l) = net(Xtrn')';
end

ypred_mlp_tst = net(Xtst')';
figure; hold on
stem(ypred_mlp_tst,'b');

R2_t.corr_svm_g_tst = corr(ypred_mlp_tst,Ytst)
R2_t.rmse_svm_g_tst = sqrt(mean((ypred_mlp_tst-Ytst).^2))
R2_t.mae_svm_g_tst  = mean(abs(ypred_mlp_tst-Ytst))

% figure(hbxplt1); hold on; boxplot(ypred_mlp_tst-Ytst);

Y1prd = ypred_mlp_tst;

% 2
Ytrn = Y2trn; Ytst = Y2tst;
mlpNit=25; mlpD=zeros(length(Ytrn),mlpNit);
net = feedforwardnet([4],'trainlm');
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
net.trainParam.showWindow = false;
net.trainParam.time = inf;
for l=1:50 %mlpNit
    [net,tr] = train(net,Xtrn',Ytrn');
    mlpD(:,l) = net(Xtrn')';
end

ypred_mlp_tst = net(Xtst')';
figure; hold on
stem(ypred_mlp_tst,'b');

R2_t.corr_svm_g_tst = corr(ypred_mlp_tst,Ytst)
R2_t.rmse_svm_g_tst = sqrt(mean((ypred_mlp_tst-Ytst).^2))
R2_t.mae_svm_g_tst  = mean(abs(ypred_mlp_tst-Ytst))

% figure(hbxplt2); hold on; boxplot(ypred_mlp_tst-Ytst);

Y2prd = ypred_mlp_tst;
Yprd = [Y1prd,Y2prd];

% writematrix(Yprd,'MLP_predMLPCA.xls');

return;