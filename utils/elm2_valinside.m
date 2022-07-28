function [ypred_train, TargetTest, TargetTrain, ypred, ValTime, TestingTime] = elm(predictores,target)

%Tasa de entrenamiento del 80%
tasa = 0.8;
n_train = round(tasa*length(predictores));
a1=randperm(length(predictores));

%Datos entrada aleatorizados
EntradaTrain = predictores(a1(1:n_train),:);
EntradaTest = predictores(a1((n_train+1):end),:);
TargetTrain = target(a1(1:n_train));
TargetTest = target(a1((n_train+1):end));

%Normailizo datos de entrenamiento (Incluya las funciones de normalizado
%implementadas en la práctica de regresión lineal del tema 5)
[EntradaTrain,Mdatos,mdatos]=normalizar(EntradaTrain,1,0,0);
[EntradaTest,~,~]=normalizar(EntradaTest,2,Mdatos,mdatos);
[TargetTrain,Mresul,mresul]=normalizar(TargetTrain,1,0,0);

%Datos y normalizado en busqueda grid (Parte de validación)
tasa = 0.2;
n_train2 = round(tasa*length(EntradaTrain));
a2=randperm(length(EntradaTrain));  %%Permutacion aleatoria de N filas en este caso
matrizEntradaTrain = EntradaTrain(a2(1:n_train2),:);
matrizTargetTrain = TargetTrain(a2(1:n_train2),:);

% [matrizEntradaTrain,Mdatos,mdatos]=normalizar(matrizEntradaTrain,1,0,0);
% [matrizTargetTrain,~,~]=normalizar(matrizTargetTrain,1,Mdatos,mdatos);

EntradaTrain(a2(1:n_train2),:) = [];
TargetTrain(a2(1:n_train2),:) = [];
i=1;
%% Busqueda grid y validacion cruzada(asi evitamos el sobreentrenamiento)
for(Nh=50:150)
    
    N=size(EntradaTrain,1);
    n=size(EntradaTrain,2);
   
    w=2*rand(n,Nh)-1;
    b=rand(1,Nh);
    g=zeros(N,Nh);

    EntradaTrain=transpose(EntradaTrain);
    %train
    for k=1:N
        g(k,:)=dot(repmat(EntradaTrain(:,k),1,Nh),w,1)+b;
    end
    %H=sin(g);
    H=1./(1+exp(-g));
    %H=(exp(g)-exp(-g))./(exp(g)+exp(-g));

    %calculo de Betas
    B=pinv(H)*TargetTrain;
    
    ypred_train=H*B;

   clear H,g,N
    N=size(matrizEntradaTrain,1);
    g=zeros(N,Nh);
    matrizEntradaTrain=transpose(matrizEntradaTrain);

    %Comienzo tiempo de validación
    tic
    
    %Calculamos producto vectorial de pesos con cada una de las variables
    %predictoras + bias (y que serán las filas de la matriz H antes de aplicar
    %la función de activación)
    for k=1:N
        g(k,:)=dot(repmat(matrizEntradaTrain(:,k),1,Nh),w,1)+b;
    end

    %Para la obtención de la matriz H usamos la función sigmoide
    H=1./(1+exp(-g));

    %Obtenemos la salida predicha de test
    ypredval=H*B;
    ValTime=toc

    valacc(i) = sqrt(sum((matrizTargetTrain-ypredval).^2)/N); %%Error cuadratico medio de validación
    
    Param(i)=Nh;
    B_model{i}=B;
    w_model{i}=w;
    b_model{i}=b;
    i=i+1;
    clear N,n,B,H,g
    EntradaTrain=transpose(EntradaTrain);
    matrizEntradaTrain=transpose(matrizEntradaTrain);
end

[x pos] = min(valacc);
Nh = Param(pos);
B = B_model{pos};
w = w_model{pos};
b = b_model{pos};


% %% Entrenamiento
% clear {N,n,B,trainacc}
% EntradaTrain(a2(1:n_train2),:) = [];
% TargetTrain(a2(1:n_train2),:) = [];
% 
% N=size(EntradaTrain,1);
% % n=size(EntradaTrain,2);
% 
% %Comienzo tiempo de entrenamiento
% tic
% %Asignación aleatoria de pesos y bias en función del número de neuronas Nh
% w=2*rand(n,Nh)-1;
% b=rand(1,Nh);
% 
% g=zeros(N,Nh);
% EntradaTrain = transpose(EntradaTrain);
% 
% %Calculamos producto vectorial de pesos con cada una de las variables
% %predictoras + bias (y que serán las filas de la matriz H antes de aplicar
% %la función de activación)
% for k=1:N
%     g(k,:) = dot(repmat(EntradaTrain(:,k),1,Nh),w,1)+b;
% end
% 
% %Para la obtención de la matriz H usamos la función sigmoide
% H=1./(1+exp(-g));
% 
% %Calculamos la pseudo inversa de Moor Penrose para la obtención del modelo
% B=pinv(H)*TargetTrain;
% 
% %Obtenemos la salida de entrenamiento
% ypred_train_n=H*B;
% 
% %finaliza tiempo de cómputo de entrenamiento
% TrainingTime=toc;
% 
% %Desnormalizo la salida de train obtenida y TargetTrain
% ypred_train = desnormalizar(ypred_train_n,1,Mresul,mresul);
% TargetTrain = desnormalizar(TargetTrain,1,Mresul,mresul);


%% Testeo
clear N,n,H,g


N=size(EntradaTest,1);
g=zeros(N,Nh);
EntradaTest=transpose(EntradaTest);

%Comienzo tiempo de entrenamiento
tic

%Calculamos producto vectorial de pesos con cada una de las variables
%predictoras + bias (y que serán las filas de la matriz H antes de aplicar
%la función de activación)
for k=1:N
    g(k,:)=dot(repmat(EntradaTest(:,k),1,Nh),w,1)+b;
end

%Para la obtención de la matriz H usamos la función sigmoide
H=1./(1+exp(-g));

%Obtenemos la salida predicha de test
ypredn=H*B;

%finaliza tiempo de cómputo de test
TestingTime=toc;

%Desnormalizo la salida de test obtenida
ypred=desnormalizar(ypredn,1,Mresul,mresul);


function [datosnorm,a,b]=normalizar(datos,tipo,a,b)

if tipo==1 %normalizacion entre -1 y +1
    b=min(datos);

    N=size(datos,1);
    m=repmat(b,N,1);
    
    datos=(datos-m);
    a=max(datos);
    
    M=repmat(a,N,1);
    
    datos=datos./M;
    datosnorm=2*datos-1;
elseif tipo==2 %normalizacion lineal max y min de entrada
    N=size(datos,1);
    m=repmat(b,N,1);
    M=repmat(a,N,1);
    
    datos=(datos-m)./M;
    datosnorm=2*datos-1;
elseif tipo==3  %normalizacion de media y varianza con obtencion de las mismas
                %media y varianza por cada caracteristica
    a=mean(datos);
    b=var(datos);
    datosnorm=(datos-repmat(a,size(datos,1),1))./sqrt(repmat(b,size(datos,1),1));
elseif tipo==4 %normalizacion de media y varianza con estas como parametros de entrada
    datosnorm=(datos-repmat(a,size(datos,1),1))./sqrt(repmat(b,size(datos,1),1));
end

function datos=desnormalizar(datosnorm,tipo,a,b)

if tipo==1
    
    datosnorm=(datosnorm+1)/2;
    
    N=size(datosnorm,1);
    
    M=repmat(a,N,1);
    m=repmat(b,N,1);    
    datos=datosnorm.*M+m;
elseif tipo==2
    datos=datosnorm.*sqrt(repmat(b,size(datosnorm,1),1))+repmat(a,size(datosnorm,1),1);
end