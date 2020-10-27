%-----------------------------------------------------
%           BackPropation por Christian Diaz
%-----------------------------------------------------
%            Entrada deseada 
%-----------------------------------------------------
x= [
    1,1,0;
    1,0,1;
    1,1,1;
    ];
%------------------------------------------------------
%           Salida deseada en binario
%------------------------------------------------------
yd = [
    1,1,1;
    1,1,0;
    0,1,1;
    ] 
%-------------------------------------------------
% Definicion de la Arquitectura Manual de la RED
%-------------------------------------------------
entradas=3;
ocultas=55;%neuronas ocultas
salidas=3;%neuronas salidas
patron=3;%patrones

alfa = 0.4; %coeficiente de aprendizaje
betha = 0.4; %Permite el Momentum

%-------------------------------------------------------------------
%                   Llamado a la funcion de la RED
%--------------------------------------------------------------------
backprogation(x,yd,entradas,ocultas,salidas,patron,alfa,betha);

%----------------------------------------------------------------------
%   Esta funcion realiza backropagation y retorna yo que se debe aproximar
%   a la salida deseada
%----------------------------------------------------------------------
function backprogation(x,yd,entradas,ocultas,salidas,patron,alfa,betha)
%-------------
%se divide para hacerlo 1
%---------------
for i=1:length(x)
    for j=1:entradas
        if(x(i,j)~=0)
            x(i,j)=x(i,j)/x(i,j);
        end
    end
end

%--------------------------------
%   Arrays para diagramar
%--------------------------------
casosVsET = [];
arrayET=[];
%---------------------------------------
%       PESOS aleatorios de las capas
%---------------------------------------
%         Entrada y de Oculta
wh = random('Normal',0,1,1,(entradas*ocultas));
th = random('Normal',0,1,1,ocultas);
%        Oculta y de salida
wo = random('Normal',0,1,1,(ocultas*salidas));
tk = random('Normal',0,1,1,salidas);
%------------------------------------------------
%        Pesos auxiliares para calculos de delta
%        se le asigna el mismo tamaño para acumular
whAux = wh;
woAux = wo;
thAux = th;
tkAux = tk;

%------------------------------------
%        Ciclo total de la red
%---------------------------------
casosEntrenamiento = 0; 

%------------------------------------
%       Errorees de la red
%---------------------------------
errorTotal = 0;
errorEsperado =0.000001;

error= [0];
errorAuxiliar=[0];

% %---------------------------------------
% %    Comienzo del entrenamiento
% %-----------------------------------------
while true
    errorTotal=0;
    errorAuxiliar=0;
    %----------------------------------
    %       Calculo por patron
    %----------------------------------
    for p=1:patron
        %-----------------------------
        %    Calculo NET ocultas
        %------------------------------
        acumulador=1;
        for j=1:ocultas
            for i=1:entradas
                if i==1
                    nethj(j)=x(p,i)*wh(acumulador)+th(j);
                else
                    nethj(j)=nethj(j)+x(p,i)*wh(acumulador);
                end
                acumulador=acumulador+1;
            end
            yhj(j) = 1/(1+exp(-nethj(j)));
        end
        %-----------------------------
        %    Calculo NET Salidas
        %------------------------------
        acumulador=1;
        for k=1:salidas
            for j=1:ocultas
                if j==1
                    netok(k)=yhj(j)*wo(acumulador)+tk(k);
                else
                    netok(k)=netok(k)+yhj(j)*wo(acumulador);
                end
                acumulador=acumulador+1;
            end
            yok(p,k) = 1/(1+exp(-netok(k)));
        end
        %--------------------------------------------
        %           BAKCPROPAGATION
        %--------------------------------------------
        %      Calculo de Delta Error Salida
        %--------------------------------------------
        for k=1:1:salidas
            dok(p,k)=(yd(p,k)-yok(p,k))*yok(p,k)*(1-yok(p,k));
        end

        %--------------------------------------------
        %      Calculo de erroresParciales Ocultas
        %--------------------------------------------
        acumulador=1;
        for k=1:salidas
            for j=1:ocultas
                for i=1:1:entradas
                    dhj(j,i)=x(p,i)*(1-x(p,i))*dok(p,k)*wo(acumulador);
                end
                acumulador=acumulador+1;
            end
        end
        %--------------------------------------------
        %      Actualización de pesos salida
        %--------------------------------------------
        acumulador=1;
        for k=1:salidas
            for j=1:ocultas
                wo(acumulador)=wo(acumulador)+alfa*dok(p,k)*yhj(j)+((wo(acumulador)-woAux(acumulador))*betha);
                acumulador=acumulador+1;
            end
            tk(k)=tk(k)+alfa*dok(p,k)+((tk(k)-tkAux(k))*betha);
        end
        woAux = wo;
        tkAux = tk;
        %--------------------------------------------
        %      Actualización de pesos entrada
        %--------------------------------------------
        acumulador=1;
        for j=1:ocultas
            for i=1:entradas
                wh(acumulador)=wh(acumulador)+alfa*dhj(j,i)*x(p,i)+((wh(acumulador)-whAux(acumulador))*betha);
                dhj(j,i)=dhj(j,i)+dhj(j,i);
                acumulador=acumulador+1;
            end
            th(j)=(th(j)+alfa*(dhj(j,entradas))/entradas)+((th(j)-thAux(j))*betha);
        end
        whAux = wh;
        thAux = th;
        %--------------------------------------------
        %      Validar el error
        %--------------------------------------------
        for k=1:salidas
            error(p,k)=0.5*(dok(p,k))^2;
        end
        
        for k=1:salidas
            errorAuxiliar=errorAuxiliar+error(p,k);
        end

        errorAuxiliar=errorAuxiliar/salidas;
        
        errorTotal=errorAuxiliar+errorTotal;
    end
    
    for p=1:patron
        for k=1:salidas
            yo(p,k)=[yok(p,k)]
        end
    end

    disp(casosEntrenamiento);
    errorTotal=errorTotal/patron;
    casosEntrenamiento=casosEntrenamiento+1;
    
    casosVsET(casosEntrenamiento)=casosEntrenamiento;
    arrayET(casosEntrenamiento)=errorTotal;
    if errorTotal<errorEsperado
        break;
    end
end
%---------------------------------------
%    fin del entrenamiento
%-----------------------------------------


pintarDiagramas(casosVsET,arrayET);
end


function pintarDiagramas (casosVsET,arrayET)

    plot(casosVsET,arrayET,'*');
    title('Casos vs Error');
    xlabel('Casos');
    ylabel('Error');
    grid on; 
    hold on;
end
