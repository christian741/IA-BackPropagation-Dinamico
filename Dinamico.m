%-----------------------------------------------------
%       Asignacion Manual de entradas neuronas y salidas
%------------------------------------------------------
CantidadEntradas = 3;
CantidadOcultas = 2;
CantidadSalidas = 2;



%Alpha = 0.0001;%Coeficiente de aprendizaje
Alpha = 0.25;
%------------------------------------------
%               UMBRALES
%------------------------------------------

CantidadUmbral =CantidadSalidas+CantidadOcultas;
UmbralOcultas= [CantidadOcultas]; % en array los umbrales
UmbralSalidas =[CantidadSalidas];
%CantidadPesos = (2*CantidadEntradas)+(2*CantidadSalidas);

%
%           Umbrales Ocultas
%
for i = 1:CantidadOcultas
    UmbralOcultas(i) = 1;
end
%
%           Umbrales Salida
%
for i = 1:CantidadSalidas
    UmbralSalidas(i) = 1;
end
%------------------------------------------
%               BIAS
%------------------------------------------

BiasOcultas= 0.5;
BiasSalidas = 0.5;

%BiasOcultas= 0;
%BiasSalidas = 0;
%------------------------------------------
%          CAPA DE ENTRADA
%------------------------------------------
Entradas = [CantidadEntradas]; %Arreglo de 1*n entradas
CantidadPesoEntrada = CantidadOcultas*CantidadEntradas;
PesosEntrada = [CantidadPesoEntrada]

%------------------------------------------
%          CAPA DE SALIDA
%------------------------------------------
Salidas = [CantidadSalidas];%Arreglo de 1*n salidas
CantidadPesoSalida = CantidadOcultas*CantidadSalidas;
PesosSalida = [CantidadPesoSalida];

NETsalida = [CantidadSalidas]; %Salida
ySalida= [CantidadSalidas];%Salidas
%
%                   ERROR
%

DeltaErrorSalida = [CantidadSalidas];%Salidas



%------------------------------------------
%                   CAPA OCULTA
%------------------------------------------
NETs =[CantidadOcultas];% Ocultas
yNeuronas = [CantidadOcultas]; %Ocultas
%
%                   ERROR
%
DeltaErroresOculta = [CantidadPesoEntrada];%Ocultas




%-----------------------------------------------------
%           Salida deseada
%----------------------------------------------------
SalidaDeseada = [CantidadSalidas];



%----------------------------------------
%           ERRORES
%-----------------------------------------
ErrorPatron = 0;
ErrorReal =0; %Error de toda la red
ErrorDefinido = 0.000000001;
%----------------------------------------
%           INTEGRACION MANUAL
%-----------------------------------------
Entradas(1) = 1;
Entradas(2) = 4;
Entradas(3) = 5;
% Entradas(1) = 0;
% Entradas(2) = 1;
%SalidaDeseada (1) =0.58778666;
% SalidaDeseada (1) =1;
SalidaDeseada (1) =0.1;
SalidaDeseada (2) =0.05;

%-------------------------------------------------------
%               PESOS
%---------------------------------------------------------
%
%     Pesos Randomicos de acuerdo a entradas
%
pesosOculta = [0.1,0.2,0.3,0.4,0.5,0.6];
pesosSalida = [0.7,0.8,0.9,0.1];
%pesosOculta = [0.1,0.5,-0.7,0.3];
%pesosSalida = [0.2,0.4];
for i = 1:CantidadPesoEntrada
    PesosEntrada(i) = pesosOculta(i);
end
%
%     Pesos Randomicos de acuerdo a Salidas
%
for i = 1:CantidadPesoSalida
    PesosSalida(i) = pesosSalida(i);
end

%**********************************************************************************************
%                               ENTRENAMIENTO
%**********************************************************************************************
par= 2;
total =0;
entrenar = false;
%**************************************************
%                       NETS OCULTAS
% Entrada a capa oculta
% w_1x_1+w_3x_2+w_5x_3+b_1=z_{h_1}
% 
% w_2x_1+w_4x_2+w_6x_3+b_1=z_{h_2}
% 
% h_1=\sigma(z_{h_1})
% 
% h_2=\sigma(z_{h_2})
%**************************************************
disp("%-------------------------------");
disp("%           NetsOcultas         ");
disp("%-------------------------------");
for i = 1:CantidadOcultas
    
    NETs(i) = 0;
    total = 0;
    %Nets de todas las neuronas capa oculta
    NETs(i)=NetsOcultas(CantidadOcultas,CantidadPesoEntrada,PesosEntrada,CantidadEntradas,Entradas,i,BiasOcultas);
    %*******************************************
    %Forward yNeuronas
    total = total+ NETs(i);
    NETs(i) = total;
    disp("NetsOcultas "+NETs(i));
    yNeuronas(i) = Sigmoideal(NETs(i));
    disp("YNeuronasOcultas "+yNeuronas(i));
    
end
%-------------------------------------------------
%           NETS de Salida
%-------------------------------------------------
disp("%-------------------------------");
disp("%           NETS de Salida         ");
disp("%-------------------------------");
for i = 1:CantidadOcultas
    NETsalida(i) = 0;
    total = 0;
    %Nets de todas las neuronas capa oculta
    NETsalida(i)=NetsSalida(CantidadSalidas,CantidadPesoSalida,PesosSalida,CantidadOcultas,yNeuronas,i,BiasSalidas)
    %*******************************************
    %Forward yNeuronas
    total = total+ NETsalida(i);
    NETsalida(i)=total;
    disp("NetsSalida "+NETsalida(i));
    ySalida(i) = Sigmoideal(NETsalida(i));
    disp("YNeuronasSalida "+yNeuronas(i));
    
end
%***********************************************************
%          CALCULO ERRORES DE DELTA SALIDA BACKPROPAGATION
%***********************************************************

for i=1:CantidadSalidas;
        
    DeltaErrorSalida(i)= (SalidaDeseada(i)-ySalida(i))*ySalida(i)*(1-ySalida(i));
    disp("DeltaErrorSalida "+DeltaErrorSalida(i));
end
%***********************************************************
%          ERRORES DE DELTA ESTIMADOS
%***********************************************************


















%*************************************************************************************
%*************************************************************************************
%*************************************************************************************
%                           FUNCIONES
%*************************************************************************************
%*************************************************************************************
%*************************************************************************************
%-------------------------------
%           NetsSalida
%-------------------------------
function valor = NetsSalida(CantidadSalidas,CantidadPesoSalida,PesosSalida,CantidadOcultas,yNeuronas,indice,BiasSalidas)

acumulador= 0;
valor= 0;
vectorAuxiliar=[];
iteracion = 1;
iteracionAux = 1;
for i=indice:CantidadSalidas:CantidadPesoSalida
    vectorAuxiliar(iteracionAux) =  PesosSalida(i);
    %disp("vector auxiliar "+vectorAuxiliar(iteracionAux) + "itearacionAux " +iteracionAux);
    iteracionAux = iteracionAux+1;
    
end
for j=1:CantidadSalidas%
    
    acumulador = yNeuronas(j)*vectorAuxiliar(iteracion);
    
    disp("i= "+iteracion+" j= "+j+ " ´´´´´´´´´´"+ yNeuronas(j)+ " * "+vectorAuxiliar(iteracion));
    iteracion=iteracion+1;
    valor = valor+acumulador;
    disp("Valor es "+valor);
end
    valor = valor+BiasSalidas;
end
%-------------------------------
%           NetsOcultas
%-------------------------------
function valor = NetsOcultas(CantidadOcultas,CantidadPesoEntrada,PesosEntrada,CantidadEntradas,Entradas,indice,BiasOcultas)

acumulador= 0;
valor= 0;
vectorAuxiliar=[];
iteracion = 1;
iteracionAux = 1;
for i=indice:CantidadOcultas:CantidadPesoEntrada
    vectorAuxiliar(iteracionAux) =  PesosEntrada(i);
%     disp("vector auxiliar "+vectorAuxiliar(iteracionAux) + "itearacionAux " +iteracionAux);
    iteracionAux = iteracionAux+1;
    
end
for j=1:CantidadEntradas%
    
    acumulador = Entradas(j)*vectorAuxiliar(iteracion);
    
    disp("i= "+iteracion+" j= "+j+ " ´´´´´´´´´´"+ Entradas(j)+ " * "+vectorAuxiliar(iteracion));
    iteracion=iteracion+1;
    valor = valor+acumulador;
%     disp("Valor es "+valor);
end
valor = valor +BiasOcultas;
end
%-------------------------------
%       Sigmoideal
%-------------------------------
function y = Sigmoideal(net)
    euler=2.71828;
    y = 1/(1+(euler^(-1*net)));
end