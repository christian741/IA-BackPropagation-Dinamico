%-----------------------------------------------------
%       Asignacion Manual de entradas neuronas y salidas
%------------------------------------------------------
CantidadEntradas = 2;
CantidadSalidas = 1;
CantidadOcultas = 2;


Alpha = 0.0001;%Coeficiente de aprendizaje
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
%----------------------------------------
%           INTEGRACION MANUAL
%-----------------------------------------
Entradas(1) = 1.5;
Entradas(2) = 1.2;
SalidaDeseada (1) =0.58778666;

%-------------------------------------------------------
%               PESOS
%---------------------------------------------------------
%
%     Pesos Randomicos de acuerdo a entradas
%
for i = 1:CantidadPesoEntrada
    PesosEntrada(i) = randi([1 100],1,1)/100;
end
%
%     Pesos Randomicos de acuerdo a Salidas
%
for i = 1:CantidadPesoSalida
    PesosSalida(i) = randi([1 100],1,1)/100;
end

%**********************************************************************************************
%                               ENTRENAMIENTO
%**********************************************************************************************
par= 2;
total =0;
%**************************************************
%Buscar Cantidad Ocultas para asignar NETS OCULTAS
%**************************************************
for i = 1:CantidadOcultas
    %Mirar si es par o Impar la Neurona y Realizar La Funcion Sigmoide
    modulo = mod(i,par);
    NETs(i) = 0;
    if(modulo == 0)
        %Es par
        total =0;
        valor = MultiParForward(CantidadPesoEntrada,PesosEntrada,CantidadEntradas,Entradas);
        total = valor +UmbralOcultas(i);
        valor =0;
    else
        %Es impar
        total =0;
        valor = MultiImparForward(CantidadPesoEntrada,PesosEntrada,CantidadEntradas,Entradas);
        total = valor +UmbralOcultas(i);
        valor =0;
    end
    %Nets de todas las neuronas capa oculta
    NETs(i)=total;
    %*******************************************
    %Forward yNeuronas
    yNeuronas(i) = Forward(NETs(i));
    
  
end
disp("ENTRADA  "+ " | Cantidad Entradas "+ length(Entradas) + " | Pesos Entradas "+ length(PesosEntrada) );
disp("--");
disp("OCULTAS | UmbralOcultas "+ length(UmbralOcultas) +" | NETS Ocultos "+ length(NETs)+ " | Yneuronas "+ length(yNeuronas) );
disp("--");
disp("Salida  "+ " | Cantidad Entradas "+ length(Salidas) + " | Pesos Salidas "+ length(PesosSalida) );
    
%**************************************************
%               NETS SALIDA
%**************************************************
for i=1:CantidadSalidas
    modulo = mod(i,par);
    NETsalida(i) = 0;
    if(modulo == 0)
        %Es par
        total =0;
        valor = MultiParForwardSalida(CantidadPesoSalida,PesosSalida,CantidadSalidas,yNeuronas);
        total = valor +UmbralSalidas(i);
        valor =0;
    else
        %Es impar
        total =0;
        valor = MultiImparForwardSalida(CantidadPesoSalida,PesosSalida,CantidadSalidas,yNeuronas);
        total = valor +UmbralSalidas(i);
        valor =0;
    end
    NETsalida(i)=total;
    ySalida(i) = Forward(NETsalida(i));
end
%***********************************************************
%          CALCULO ERRORES DE DELTA SALIDA BACKPROPAGATION
%***********************************************************

for i=1;CantidadSalidas;
    VectorAuxiliar = [];
    VectorAuxiliar2 = [];
    
    DeltaErrorSalida(i)= (SalidaDeseada(i)-ySalida(i))*ySalida(i)*(1-ySalida(i));
end

%***********************************************************
%          ERRORES DE DELTA ESTIMADOS
%***********************************************************



%     for k=1:CantidadSalidas %1
%         for j=1:CantidadPesoSalida %2
%             valor = Entradas(j)*(1-Entradas(j));
%             total = total + valor *(DeltaErrorSalida(k)*UmbralOcultas(j));
%         end
%     end
%     DeltaErroresOculta(i) = total;
% valores =[];
% totalErrorParcial = ParcialError(DeltaErrorSalida);
% valores (j) = Entradas(j)*(1-Entradas(j))*totalErrorParcial;
% DeltaErroresOculta(i,j)=[i,valores(j)];
for i=1;CantidadPesoEntrada
    hecho =false;
    j= 0;
    while hecho==false && j<CantidadPesoEntrada
        j=1;
        vectorAE =[];
        for k=1:CantidadEntradas
            vectorAE(k) = Entradas(k)*(1-Entradas(k));
        end
        vectorPS(j) = ParcialError(DeltaErrorSalida);
        j=j+1;
    end
end





















%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%                                         BACK ERRORES
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function vectorPS = ParcialError (DeltaErrorSalida)
    valor =0;
    total =0;
    vectorPS = [];
    for j=1:CantidadPesoSalida%2
        for i=1;CantidadSalidas%1
            vectorPS(j) = DeltaErrorSalida(i)* PesosSalida(j);
        end
    end
end



%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%                                         FORWARD SALIDAS
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%-------------------------------
%           MitplicarImparSalidas
%-------------------------------
function valor = MultiImparForwardSalida(CantidadPesoSalida,PesosSalida,CantidadSalidas,yNeuronas)
    
    vectAuxiliarIP = [];%Pesos Impares
    vactAuxiliarE = [];%Entradas
    valorAuxiliar = 0;
    valor =0;
    for i=1:CantidadPesoSalida
        if(mod(i,2)~=0) 
        vectAuxiliarIP(i)=PesosSalida(i);
        end
    end
    for i=1:length(yNeuronas)
        vectAuxiliarE(i)=yNeuronas(i);
    end
    for i=1:CantidadSalidas
        valorAuxiliar = vectAuxiliarIP(i)*vectAuxiliarE(i);
        valor = valor+valorAuxiliar;
    end
    
end


%-------------------------------
%           MitplicarParSalida
%-------------------------------
function valor = MultiParForwardSalida(CantidadPesoSalida,PesosSalida,CantidadSalidas,yNeuronas)
    
      vectAuxiliarIP = [];%Pesos Impares
    vactAuxiliarE = [];%Entradas
    valorAuxiliar = 0;
    valor =0;
    for i=1:CantidadPesoSalida
        if(mod(i,2)==0)
        vectAuxiliarIP(i)=PesosSalida(i);
        end
    end
    for i=1:CantidadSalidas
        vectAuxiliarE(i)=yNeuronas(i);
    end
    for i=1:CantidadSalidas
        valorAuxiliar = vectAuxiliarIP(i)*vectAuxiliarE(i);
        valor = valor+valorAuxiliar;
    end
    
end
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%                                         FORWARD OCULTAS
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%-------------------------------
%           MitplicarImpar
%-------------------------------
function valor = MultiImparForward(CantidadPesoEntrada,PesosEntrada,CantidadEntradas,Entradas)
    
    vectAuxiliarIP = [];%Pesos Impares
    vactAuxiliarE = [];%Entradas
    valorAuxiliar = 0;
    valor =0;
    for i=1:CantidadPesoEntrada
        if(mod(i,2)~=0)
        vectAuxiliarIP(i)=PesosEntrada(i);
        end
    end
    for i=1:CantidadEntradas
        vectAuxiliarE(i)=Entradas(i);
    end
    for i=1:CantidadEntradas
        
        valorAuxiliar = vectAuxiliarIP(i)*vectAuxiliarE(i);
        valor = valor+valorAuxiliar;
        
    end
    
end


%-------------------------------
%           MitplicarPar
%-------------------------------
function valor = MultiParForward(CantidadPesoEntrada,PesosEntrada,CantidadEntradas,Entradas)
    
    vectAuxiliarP = [];%Pesos Pares
    vactAuxiliarE = [];%Entradas
    valorAuxiliar = 0;
    valor =0;
    for i=1:CantidadPesoEntrada
        if(mod(i,2)==0)
        vectAuxiliarP(i)=PesosEntrada(i);
        end
    end
    for i=1:CantidadEntradas
        vectAuxiliarE(i)=Entradas(i);
    end
    for i=1:CantidadEntradas
        valorAuxiliar = vectAuxiliarP(i)*vectAuxiliarE(i);
        valor = valor+valorAuxiliar;
    end
    
end
%-------------------------------
%       FORWARD
%-------------------------------
function y = Forward(net)
    euler=2.71828;
    y = 1/(1+(euler^(-1*net)));
end