function [ypred_train, TargetTrain, ypred, TrainingTime, TestingTime] = elm(EntradaTrain,EntradaTest,TargetTrain,TargetTest,elmS)



%Normailizo datos de entrenamiento 
[EntradaTrain,Mdatos,mdatos]=normalizar(EntradaTrain,1,0,0);
[EntradaTest,~,~]=normalizar(EntradaTest,2,Mdatos,mdatos);
[TargetTrain,Mresul,mresul]=normalizar(TargetTrain,1,0,0);

%Datos y normalizado en busqueda grid (Parte de validación)
tasa = 0.2;
n_train2 = round(tasa*length(EntradaTrain));
a2=randperm(length(EntradaTrain));  %%Permutacion aleatoria de N filas en este caso
matrizEntradaTrain = EntradaTrain(a2(1:n_train2),:); %Conjunto de validacion
matrizTargetTrain = TargetTrain(a2(1:n_train2),:); %Target de validacion

EntradaTrain(a2(1:n_train2),:) = []; %Conjunto de entrenamiento
TargetTrain(a2(1:n_train2),:) = []; %Target de entrenamiento
i=1;
NhInit = elmS.NhInit; NhEnd = elmS.NhEnd;
%% Entrenamiento y validación
tic
for Nh=NhInit:NhEnd
    
    N=size(EntradaTrain,1);
    n=size(EntradaTrain,2);
   
    w=2*rand(n,Nh)-1;
    b=rand(1,Nh);
    g=zeros(N,Nh);

    EntradaTrain=transpose(EntradaTrain);
  
    for k=1:N
        g(k,:)=dot(repmat(EntradaTrain(:,k),1,Nh),w,1)+b;
    end
    %H=sin(g);
    H=1./(1+exp(-g));
    %H=(exp(g)-exp(-g))./(exp(g)+exp(-g));

    % Modelo de entrenamiento
    B=pinv(H)*TargetTrain;
    
    ypred_train=H*B;

    clear {H,g,N};
    N=size(matrizEntradaTrain,1);
    g=zeros(N,Nh);
    matrizEntradaTrain=transpose(matrizEntradaTrain);

 
    %% Valiacion
    
    for k=1:N
        g(k,:)=dot(repmat(matrizEntradaTrain(:,k),1,Nh),w,1)+b;
    end

    %Para la obtención de la matriz H usamos la función sigmoide
    H=1./(1+exp(-g));

    %Obtenemos la salida predicha de validacion
    ypredval=H*B;
    

    valacc(i) = sqrt(sum((matrizTargetTrain-ypredval).^2)/N); %%Error cuadratico medio de validación
    
    Param(i)=Nh;
    B_model{i}=B;
    w_model{i}=w;
    b_model{i}=b;
    i=i+1;
    clear {N,n,B,H,g};
    EntradaTrain=transpose(EntradaTrain);
    matrizEntradaTrain=transpose(matrizEntradaTrain);
end
TrainingTime=toc;

[~, pos] = min(valacc);
Nh = Param(pos);
B = B_model{pos};
w = w_model{pos};
b = b_model{pos};


%% Testeo
clear {N,n,H,g};


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