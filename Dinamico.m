% -------------------------
%               RNABP Dinamica
% -------------------------
%-----------------------------------------------------
%       Definicion de Entradas y Salidas Deseadas
%------------------------------------------------------
% X=[0 0 255;
%    0 255 0;
%    255 0 0] ; %Entradas
% YD =[1 0 0;
%      0 1 0; 
%      1 0 0] ; %SalidasDeseadas
% X=[0 0 0; 255 255 255]
% YD =[1; 0]
X=[1.5 1.2];
YD=[0.69];
% x=[0 0;
%    0 1;
%    1 0;
%    1 1];
% yd=[0; 1; 1; 1];

%-----------------------------------------------------
%       Asignacion Manual de entradas neuronas y salidas
%------------------------------------------------------
cantidadEntradas = length(X);
cantidadOcultas = 2;
cantidadSalidas =length(YD);
%-----------------------------------------------------
%                   PESOS DE
%------------------------------------------------------
%----------------   OCULTAS    -------------------
cantidadpesosOcultas = cantidadEntradas*cantidadOcultas;
pesosOcultas = [];
pesosOcultas = [0.3 0.7 ;0.5 0.9];
%for i=1:cantidadEntradas
    %for j=1:cantidadOcultas
     %  pesosOcultas(i,j)=randi([1 100],1,1)/100;
    %end
%end

%----------------   SALIDAS    -------------------
cantidadpesosSalidas = cantidadOcultas * cantidadSalidas;
pesosSalidas = [];
for i=1:cantidadSalidas
    for j=1:cantidadOcultas
        %pesosSalidas(i,j)=randi([1 100],1,1)/100;
    end
end
pesosSalidas =[0.1  0.4]
SalidaDeseada = YD;

%-----------------------------------------------------
%                BIAS
%------------------------------------------------------
%----------------   OCULTAS    -------------------
biasOcultas = [];
for i=1:cantidadOcultas
    biasOcultas(i)=0;
end
%----------------   SALIDAS    -------------------
biasSalidas = [];
for i=1:cantidadSalidas
    biasSalidas(i,j)=0;
end
biasOcultas = [-0.6 -0.8];
biasSalidas = [-0.1];
%-----------------------------------------------------
%                    NETS DE
%------------------------------------------------------
%----------------   OCULTAS    -------------------
NETOculta = [];
for i=1:cantidadOcultas
    NETOculta(i) = 0; 
end

%----------------   SALIDAS    -------------------
NETSalida = [];
for i=1:cantidadSalidas
    NETSalida(i) = 0; 
end
%-----------------------------------------------------
%                    Y con SIGMOIDEAL DE
%------------------------------------------------------
%----------------   OCULTAS    -------------------
yOculta = [];

%----------------   SALIDAS    -------------------
ySalida = [];

%-----------------------------------------------------
%                    DELTAS DE
%------------------------------------------------------
%----------------   OCULTAS    -------------------
deltaOculta = [];
for i=1:1cantidadOcultas
    for j=1:1:cantidadEntradas
        deltaOculta(i,j)=0;
    end
end
%----------------   SALIDAS    -------------------
deltaSalida = [];
for i=1:1:cantidadOcultas
    for j=1:1:cantidadSalidas
        deltaSalida(i,j) = 0;
    end
end
%-----------------------------------------------------
%                  DELTA Error DE SALIDA
%------------------------------------------------------
DeltaErrorSalida = [];
for i=1:1cantidadOcultas
    for j=1:1:cantidadSalidas
        DeltaErrorSalida(i,j)=0;
    end
end
%-----------------------------------------------------
%                  Errores
%------------------------------------------------------
for i=1:cantidadEntradas
    errores(i)=0;
end
arrayErrorTotal = [];
%-----------------------------------------------------
%                  DEFINICION DE VARIABLES
%------------------------------------------------------
alfa=0.8;
errorTotal=100;
casos=0;
errorDeseado=0.0000001;

%while errorTotal>errorDeseado
    for p=1:1:1
        
        for i=1:1:cantidadOcultas %Filas
            for j=1:1:cantidadEntradas %Columnas
                NETOculta(p,i)=NETOculta(p,i)+pesosOcultas(j,i)*X(p,j);
                %disp("****NETOculta(p,j) "+NETOculta(p,i)+ "   i "+i+"  j "+j+"  pesosOcultas "+pesosOcultas(j,i)+"*  X "+X(p,j));
            end
        end
        for j=1:1:cantidadOcultas
            NETOculta(p,j)=NETOculta(p,j)+biasOcultas(j);
            yOculta(p,j)=1/(1+exp(-NETOculta(p,j)));
            %disp(" NETOculta "+NETOculta(p,j));
            %disp(" yOculta "+ yOculta(p,j));
        end
        
        
        for j=1:1:cantidadSalidas
            for i=1:1:cantidadOcultas
                %disp("Fila "+j+"  Columna "+i);
                NETSalida(p,j)=NETSalida(p,j)+pesosSalidas(j,i)*yOculta(p,i);
                %disp("*")
                %disp("****NETSalida(p,i) "+NETSalida(p,j)+ "   i "+i+"  j "+j+"  pesosSalidas "+pesosSalidas(j,i)+"*  yOculta "+yOculta(p,i));
                %ySalida(p,i)=1/(1+exp(-neto(p,1)));
            end
        end
        
        for j=1:1:cantidadSalidas
            NETSalida(p,j)=NETSalida(p,j)+biasSalidas(j);
            ySalida(p,j)=1/(1+exp(-NETSalida(p,j)));
            %disp(" NETSalida "+NETSalida(p,j));
            %disp(" ySalida "+ ySalida(p,j));
        end
        
        
         %calculo errores parciales
        for k=1:1:cantidadSalidas
            deltaSalida(p,k)=(YD(k)-ySalida(p,k))*ySalida(p,k)*(1-ySalida(p,k));
        end
        
        for j=1:1:cantidadOcultas
            for k=1:1:cantidadSalidas
                %disp("Fila "+j+"  Columna "+k);
                DeltaErrorSalida(p,j)=deltaSalida(p,k)*pesosOcultas(k,j);
                disp("---DeltaErrorSalida(p,i) "+DeltaErrorSalida(p,j)+ "   j "+j+"  k "+k+"  deltaSalida "+deltaSalida(p,k)+"*  pesosOcultas "+pesosOcultas(k,j));
            end
        end
        for j=1:1:cantidadOcultas
            for i=1:1:cantidadEntradas
                %disp("Fila "+i+"  Columna "+j);
                deltaOculta(p,j)=X(p,i)*(1-X(p,i))*DeltaErrorSalida(p,j);
                 disp("+++++++++deltaOculta(p,i) "+deltaOculta(p,j)+ "   j "+j+"  k "+k+"  Entradas "+X(p,i)+"*  1-eNTRDA "+X(p,i)+"  DeltaErrorSalida  "+DeltaErrorSalida(p,j));
            end
        end
        
         % actualizacion de pesos out
        for j=1:1:2
            for k=1:1:1
                wo(k,j)=wo(k,j)+alfa*Do(p,k)*yh(p,j);
                tk(k)=tk(k)+alfa*Do(p,k);
            end
        end
        
        
    end
    casos=casos+1
    arrayErrorTotal(casos)=errorTotal;
    arrayCasos(casos)= casos;
%end



















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
for j=1:CantidadOcultas%
    
    acumulador = yNeuronas(j)*vectorAuxiliar(iteracion);
    
    disp("i= "+iteracion+" j= "+j+ " ´´´´´´´´´´"+ yNeuronas(j)+ " * "+vectorAuxiliar(iteracion));
    iteracion=iteracion+1;
    valor = valor+acumulador;
    disp("Valor es "+valor);
end
    valor = valor+BiasSalidas(indice);
end
%-------------------------------
%           NetsOcultas
%-------------------------------
function valor = NetsOcultas(CantidadOcultas,CantidadPesoEntrada,PesosEntrada,CantidadEntradas,Entradas,indice,BiasOcultas,patron)

acumulador= 0;
valor= 0;
vectorAuxiliar=[];
iteracion = 1;
iteracionAux = 1;
for i=indice:CantidadOcultas:CantidadPesoEntrada
   vectorAuxiliar(iteracionAux) =  PesosEntrada(i);
   disp("vector auxiliar "+vectorAuxiliar(iteracionAux) + "itearacionAux " +iteracionAux);
    iteracionAux = iteracionAux+1;
    
end
for j=1:CantidadEntradas%
    
    acumulador = Entradas(j)*vectorAuxiliar(iteracion);
    
    disp("i= "+iteracion+" j= "+j+ " ´´´´´´´´´´"+ Entradas(j)+ " * "+vectorAuxiliar(iteracion));
    iteracion=iteracion+1;
    valor = valor+acumulador;
%     disp("Valor es "+valor);
end
valor = valor +BiasOcultas(indice);
end
%-------------------------------
%       Sigmoideal
%-------------------------------
function y = Sigmoideal(net)
    y = 1/(1+(exp(-net)));
end