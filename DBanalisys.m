%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Data analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear all;close all;

% Read Data
addpath('MLP'); addpath('utils'); addpath('data');
dtTrn = xlsread('ANN_database_training');
Xtrn  = dtTrn(:,1:end-2); Y1trn  = dtTrn(:,end-1); Y2trn  = dtTrn(:,end);
dtTst = xlsread('ANN_database_tests');
Xtst  = dtTst(:,1:end);

% Training;
rng(1);

%% Train and Test RFs
% 1
Ytrn = Y1trn;
Mdl = TreeBagger(100,Xtrn,Ytrn,'Method','regression');
ypred_rf_tst = predict(Mdl,Xtst);

figure; hold on
stem(ypred_rf_tst,'b');

Y1prd_rf = ypred_rf_tst;

% 2
Ytrn = Y2trn;
Mdl = TreeBagger(100,Xtrn,Ytrn,'Method','regression');
ypred_rf_tst = predict(Mdl,Xtst);

figure; hold on
stem(ypred_rf_tst,'b');

Y2prd_rf = ypred_rf_tst;

Yprd_rf = [Y1prd_rf,Y2prd_rf];

writematrix(Yprd_rf,'RF_pred.xls');
%% Train and Test Linear Regressors
% 1
Ytrn = Y1trn;
B_r = regress(Ytrn,Xtrn);
[B_lss,FitInfo] = lasso(Xtrn,Ytrn,'CV',10);
    figure;
    lassoPlot(B_lss,FitInfo,'PlotType','CV');
    legend('show') % Show legend
B_lss=B_lss(:,FitInfo.IndexMinMSE);
% B_rdg = ridge(Xtrn,Ytrn);

ypred_svm_lr_tst = Xtst*B_r;
ypred_svm_ls_tst = Xtst*B_lss;


figure; hold on
stem(ypred_svm_lr_tst,'b');
stem(ypred_svm_ls_tst,'g');
% stem(ypred_svm_rg_tst,'m');

Y1prd_lr = ypred_svm_lr_tst;
Y1prd_ls = ypred_svm_ls_tst;
% Y1prd_rg = ypred_svm_rg_tst;

% 2
Ytrn = Y2trn;
B_r = regress(Ytrn,Xtrn);
[B_lss,FitInfo] = lasso(Xtrn,Ytrn,'CV',10);
    figure;
    lassoPlot(B_lss,FitInfo,'PlotType','CV');
    legend('show') % Show legend
B_lss=B_lss(:,FitInfo.IndexMinMSE);
% B_rdg = ridge(Xtrn,Ytrn);

ypred_svm_lr_tst = Xtst*B_r;
ypred_svm_ls_tst = Xtst*B_lss;


figure; hold on
stem(ypred_svm_lr_tst,'b');
stem(ypred_svm_ls_tst,'g');
% stem(ypred_svm_rg_tst,'m');

Y2prd_lr = ypred_svm_lr_tst;
Y2prd_ls = ypred_svm_ls_tst;
% Y2prd_rg = ypred_svm_rg_tst;

Yprd_lr = [Y1prd_lr,Y2prd_lr];
Yprd_ls = [Y1prd_ls,Y2prd_ls];

writematrix(Yprd_lr,'LREG_pred.xls');
writematrix(Yprd_ls,'LASSO_pred.xls');

return;
%% Train and Test SVRs
% 1
Ytrn = Y1trn;

SVMModel_g = fitrsvm(Xtrn,Ytrn,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
SVMModel_l = fitrsvm(Xtrn,Ytrn,'KernelFunction','linear','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
SVMModel_p = fitrsvm(Xtrn,Ytrn,'KernelFunction','polynomial','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');

% test
ypred_svm_g_tst = predict(SVMModel_g,Xtst);
ypred_svm_l_tst = predict(SVMModel_l,Xtst);
ypred_svm_p_tst = predict(SVMModel_p,Xtst);

figure; hold on
stem(ypred_svm_g_tst,'b');
stem(ypred_svm_l_tst,'g');
stem(ypred_svm_p_tst,'m');

Y1prd_g = ypred_svm_g_tst;
Y1prd_l = ypred_svm_l_tst;
Y1prd_p = ypred_svm_p_tst;

% 2
Ytrn = Y2trn;

SVMModel_g = fitrsvm(Xtrn,Ytrn,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
SVMModel_l = fitrsvm(Xtrn,Ytrn,'KernelFunction','linear','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
SVMModel_p = fitrsvm(Xtrn,Ytrn,'KernelFunction','polynomial','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');

% test
ypred_svm_g_tst = predict(SVMModel_g,Xtst);
ypred_svm_l_tst = predict(SVMModel_l,Xtst);
ypred_svm_p_tst = predict(SVMModel_p,Xtst);

figure; hold on
stem(ypred_svm_g_tst,'b');
stem(ypred_svm_l_tst,'g');
stem(ypred_svm_p_tst,'m');

Y2prd_g = ypred_svm_g_tst;
Y2prd_l = ypred_svm_l_tst;
Y2prd_p = ypred_svm_p_tst;

Yprd_g = [Y1prd_g,Y2prd_g];
Yprd_l = [Y1prd_l,Y2prd_l];
Yprd_p = [Y1prd_p,Y2prd_p];

writematrix(Yprd_g,'SVR_g_pred.xls');
writematrix(Yprd_l,'SVR_l_pred.xls');
writematrix(Yprd_p,'SVR_p_pred.xls');

return
%% Train and Test MLPs
% 1
Ytrn = Y1trn;
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

Y1prd = ypred_mlp_tst;

% 2
Ytrn = Y2trn;
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

Y2prd = ypred_mlp_tst;
Yprd = [Y1prd,Y2prd];

writematrix(Yprd,'MLP_pred.xls');


return;
%% Train and Test SVMs
% for k=1:1
%     Y  = dt.KFva(:,:,k);
%     Ytrn  = dt.KFva(:,:,k); Ytrn=Ytrn(idTrn)';
%     Ytst  = dt.KFva(:,:,k); Ytst=Ytst(idxTst)';
%     % train
%     SVMModel_g = fitrsvm(Xtrn,Ytrn,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
%     SVMModel_l = fitrsvm(Xtrn,Ytrn,'KernelFunction','linear','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
%     SVMModel_p = fitrsvm(Xtrn,Ytrn,'KernelFunction','polynomial','KernelScale','auto','Standardize',true);%,'OptimizeHyperparameters','auto');
% 
%     % test
%     ypred_svm_g_tst = predict(SVMModel_g,Xtst);
%     ypred_svm_l_tst = predict(SVMModel_l,Xtst);
%     ypred_svm_p_tst = predict(SVMModel_p,Xtst);
%     
%     % Metrics
%     R2_t.corr_g = corr(ypred_svm_g_tst,Ytst)
%     R2_t.corr_l = corr(ypred_svm_l_tst,Ytst)
%     R2_t.corr_p = corr(ypred_svm_p_tst,Ytst)
%     R2_t.rmse_g = sqrt(mean((ypred_svm_g_tst-Ytst).^2))
%     R2_t.rmse_l = sqrt(mean((ypred_svm_l_tst-Ytst).^2))
%     R2_t.rmse_p = sqrt(mean((ypred_svm_p_tst-Ytst).^2))
%     R2_t.mae_g  = mean(abs(ypred_svm_g_tst-Ytst))
%     R2_t.mae_l  = mean(abs(ypred_svm_l_tst-Ytst))
%     R2_t.mae_p  = mean(abs(ypred_svm_p_tst-Ytst))
%     
%     figure; hold on
%     stem(abs(ypred_svm_g_tst-Ytst),'b');
%     stem(abs(ypred_svm_l_tst-Ytst),'g');
%     stem(abs(ypred_svm_p_tst-Ytst),'m');
%     
%     % test
%     ypred_svm_g_tst = predict(SVMModel_g,[X1(:),X2(:)]);
%     ypred_svm_l_tst = predict(SVMModel_l,[X1(:),X2(:)]);
%     ypred_svm_p_tst = predict(SVMModel_p,[X1(:),X2(:)]);
%     
%     figure;
%     h = gca; surf(X1,X2,abs(Y-reshape(ypred_svm_g_tst,size(X1)))); set(h,'xscale','log');
%     xlabel('r_m'); ylabel('r_\omega'); yticks(0.1:0.2:1.5); xticks([1e-4 1e-3 1e-2 1e-1 1e0]);
% %     print(sprintf('figs/k%d.eps',k),'-depsc','-painters'); savefig(sprintf('figs/k%d.fig',k));
%     colormap(jet);colorbar;
%     figure;
%     h = gca; surf(X1,X2,abs(Y-reshape(ypred_svm_l_tst,size(X1)))); set(h,'xscale','log');
%     xlabel('r_m'); ylabel('r_\omega'); yticks(0.1:0.2:1.5); xticks([1e-4 1e-3 1e-2 1e-1 1e0]);
% %     print(sprintf('figs/k%d.eps',k),'-depsc','-painters'); savefig(sprintf('figs/k%d.fig',k));
%     colormap(jet);colorbar;
%     figure;
%     h = gca; surf(X1,X2,abs(Y-reshape(ypred_svm_p_tst,size(X1)))); set(h,'xscale','log');
%     xlabel('r_m'); ylabel('r_\omega'); yticks(0.1:0.2:1.5); xticks([1e-4 1e-3 1e-2 1e-1 1e0]);
% %     print(sprintf('figs/k%d.eps',k),'-depsc','-painters'); savefig(sprintf('figs/k%d.fig',k));
%     colormap(jet);colorbar;
% end

%% Train and Test ELMs
% elmS.NhInit = 50; elmS.NhEnd = 150; elmNit=1; elmD=zeros(length(Y),elmNit);
% for l=1:elmNit
%     Y  = dt.KFva(:,:,1);
%     Ytrn  = dt.KFva(:,:,k); Ytrn=Ytrn(idTrn)';
%     Ytst  = dt.KFva(:,:,k); Ytst=Ytst(idxTst)';
%     [ypred_elm_tst, ~, ypred_elm_tst, ~, ~] = elm(Xtrn,Xtst,Ytrn,Ytst,elmS);
%     
%     R2_t.corr_elm_tst = corr(ypred_elm_tst,Ytst)
%     R2_t.rmse_elm_tst = sqrt(mean((ypred_elm_tst-Ytst).^2))
%     R2_t.mae_elm_tst  = mean(abs(ypred_elm_tst-Ytst))
%     
%     figure; hold on
%     stem(abs(ypred_elm_tst-Ytst),'b');
% 
%     % test
%     [ypred_elm_tst, ~, ypred_elm_tst, ~, ~] = elm(Xtrn,[X1(:),X2(:)],Ytrn,Y,elmS);
%     
%     figure;
%     h = gca; surf(X1,X2,abs(Y-reshape(ypred_elm_tst,size(X1)))); set(h,'xscale','log');
%     xlabel('r_m'); ylabel('r_\omega'); yticks(0.1:0.2:1.5); xticks([1e-4 1e-3 1e-2 1e-1 1e0]);
% %     print(sprintf('figs/k%d.eps',k),'-depsc','-painters'); savefig(sprintf('figs/k%d.fig',k));
%     colormap(jet);colorbar;
% end

%% Train and Test RFs
for k=1:1
    Y  = dt.KFva(:,:,1);
    Ytrn  = dt.KFva(:,:,k); Ytrn=Ytrn(idTrn)';
    Ytst  = dt.KFva(:,:,k); Ytst=Ytst(idxTst)';
    Mdl = TreeBagger(100,Xtrn,Ytrn,'Method','regression');
    
    ypred_rf_tst = predict(Mdl,Xtst);
    
    R2_t.corr_rf_tst = corr(ypred_rf_tst,Ytst)
    R2_t.rmse_rf_tst = sqrt(mean((ypred_rf_tst-Ytst).^2))
    R2_t.mae_rf_tst  = mean(abs(ypred_rf_tst-Ytst))
    
    figure; hold on
    stem(abs(ypred_rf_tst-Ytst),'b');

    % test
    ypred_rf_tst = predict(Mdl,[X1(:),X2(:)]);
    
    figure;
    h = gca; surf(X1,X2,abs(Y-reshape(ypred_rf_tst,size(X1)))); set(h,'xscale','log');
    xlabel('r_m'); ylabel('r_\omega'); yticks(0.1:0.2:1.5); xticks([1e-4 1e-3 1e-2 1e-1 1e0]);
%     print(sprintf('figs/k%d.eps',k),'-depsc','-painters'); savefig(sprintf('figs/k%d.fig',k));
    colormap(jet);colorbar;
    figure;
    h = gca; surf(X1,X2,reshape(ypred_rf_tst,size(X1))), set(h,'xscale','log');
    xlabel('r_m'); ylabel('r_\omega'); yticks(0.1:0.2:1.5); xticks([1e-4 1e-3 1e-2 1e-1 1e0]);
%     print(sprintf('figs/k%d.eps',k),'-depsc','-painters'); savefig(sprintf('figs/k%d.fig',k));
    colormap(jet);colorbar;
end
