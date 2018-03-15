# Libreria requerida! para manejo de matrices y vectores
import numpy as np
#
## @author Josepablo Cruz Baas
## @Date Lunes, 4 de diciembre 2017
#
##
### Descripcion del codigo
##
# Esta clase permite crear una red neuronal de n capas con m salidas.
# n=0 es la capa de entrada y n=n es la capa de salida.
# Implementa el algoritmo Backpropagation para la función de
#   activación logistica sigmoidal.
# Además permite el uso individual de las funciones sigmoidal y
#   Neuron para uso tipo libería
#
#
class NeuralNetwork:
    #Numero de capas
    layers = 2
    #Numero de neuronas por capa, asociada a cada capa segun indice
    neurons = []
    #Arreglo de matrices de pesos entre cada capa n y n+1
    W = []
    # Funcion sigmoidal - logistica
    def sigmoid(self,x):
        return 1/(1+(np.exp(-x)))
    
    # # 
    # A - Datos de entrada
    # W - Pesos, W[0] debe ser el peso del Bias
    # B - Bias
    #   Funcion -> Sigmoid
    def Neuron(self,A,W,B=1):
        #El primer element de los pesos es del bias
        return self.sigmoid( np.dot(A,W[1:])+(B*W[0]) )
    
    # #
    # Constructor de la clase
    #   La clase acepta un tamaño minimo de 2 capas a una neurona por capa,
    #   esto implica un perceptron de una sola entrada y una sola salida
    #layers - Numero de capas de la red
    #neurons - Arreglo ordenado de numero de neurona
    #   por capa asociados por indices
    def __init__(self,layers=2,neurons=[1,1]):
        self.layers = layers
        self.neurons = neurons
        self.W = []
        #Por cada capa
        self.W.append([0])
        for i in range(1,layers):
            #(Neuronas por capa) x (Pesos por cada neurona + Bias)
            #print(i)
            temp = np.ones( shape=(neurons[i] , neurons[i-1]+1) )
            self.W.append(temp)
    
    # #
    # Evaluacion en la capa.
    # Data - Arreglo de entradas de la red del mismo
    #    tamaño espeficicado en neurons[0]
    def Evaluate(self,Data=[1]):
        d = Data
        for l in range(1,self.layers):
            ans = np.ones(self.neurons[l])
            #print("Capa ",l,d,ans)
            for n in range(self.neurons[l]):
                #print("Neurona ",n)
                ans[n] = ( self.Neuron( d , self.W[l][n] ) )
            d = ans
        return d
    # #
    # Funcion de aprendizaje- Backpropagation o retropropagacion
    # Por defecto se elige un parametro de salto alpha = 0.001
    # NNInput - Arreglo entrada del sistema, del mismo tamanio especificado en neurons[0]
    # NNOutput - Arreglo de salida esperada del sistema, del mismo tamanio especificado en neurons[-1]
    # alpha - Parametro de "salto" de aprendizaje
    #
    def Learn(self,NNInput=[1],NNOutput=[1],alpha=0.001):
        #Hacer la evaluacion de la entrada y almacenar
            #las salidas de cada neurona
        out = []
        out.append(NNInput)
        
        d = NNInput
        for l in range(1,self.layers):
            ans = np.ones(self.neurons[l])
            #print("Capa ",l,d,ans)
            for n in range(self.neurons[l]):
                #print("Neurona ",n)
                ans[n] = ( self.Neuron( d , self.W[l][n] ) )
            d = ans
            out.append(ans)
        
        #Obtener el termino de error de cada neurona de salida k
        Dk = np.ones( self.neurons[-1] )
        for k in range(self.neurons[-1]):
            Dk[k]=( (NNOutput[k] - out[-1][k])*(1 - out[-1][k])*(out[-1][k]) ) 
            
        
        #Obtener el termino de error de cada neurona oculta
            #por capa en orden inverso. (Ultima a primera)
        Dh = []
        #Copiamos los valores de Dk anteriories, es decir, error delta de las neuronas finales
        dk = Dk
        #Por cada capa excepto la primera, porque la primer capa es la entrada
            #y no tiene termino de error
        #Y tampoco la ultima porque es la capa de salida y ha sido calculada antes
        for i in range(1,self.layers-1):
            #print(dk)
            layer = (self.layers-i)-1
            temp = np.ones( self.neurons[layer] )
            #Por cada neurona
            #print("Capa anterior ",layer+1)
            for neuron in range( self.neurons[layer] ):
                #Calcular el termino de error delta
                #print("Capa ",layer," Neurona ",neuron," neuronas ",self.neurons[layer])
                Oh = out[layer][neuron]
                Oh *= (1-Oh)
                Wk = 0
                for j in range( self.neurons[layer+1] ):
                    #Peso de la neurona j de la capa superior conectada a
                        #la neurona k de la capa actual
                    #Por el error de la neurona j de la capa superior
                    Wk += self.W[layer+1][j][neuron+1]*dk[j]
                temp[neuron] = Wk*Oh
            Dh.append(temp)
            #Actualizamos los terminos dk superiores por los nuevos dk
            dk = Dh[-1]
        
        #Actualizamos los pesos de la red
            #Pesos finales
        for neuron in range( self.neurons[-1] ):
            for k in range( self.neurons[-2] ):
                self.W[-1][neuron][k+1] += (alpha*out[-2][k])*Dk[neuron]
            self.W[-1][neuron][0] += alpha*Dk[neuron]#Actualizar el bias
            
            #Pesos ocultos, excepto primer capa y ultima
        for layer in range(1,self.layers-1):#Capa
            #print("Capa ",layer,self.neurons[layer])
            for neuron in range( self.neurons[layer] ):#Neurona de capa
                #print(Dh[-layer][neuron],layer-1,neuron)
                for k in range( self.neurons[layer-1] ):#Neurona de capa inferior
                    #print(self.W[layer][neuron][k+1])
                    self.W[layer][neuron][k+1] += (alpha*out[layer-1][k])*Dh[-layer][neuron]
                    #print(self.W[layer][neuron][k+1])
                #Actualizacion del bias!
            for k in range( self.neurons[layer] ):
                    self.W[layer][neuron][0] += alpha*Dh[-layer][k]
