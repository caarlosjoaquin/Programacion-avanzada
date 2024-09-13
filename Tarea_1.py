import math
import numpy as np
import time
from time import perf_counter
import matplotlib.pyplot as plt


tiempos_combinatoria = []
tiempos_programacion_dinamica = []
tiempos_memoizacion = []

def medir_tiempo(func):
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        tiempo = fin - inicio
        #print(tiempo)
        #print(f"Tiempo de ejecución de {func.__name__}: {fin - inicio:.6f} segundos")
        return tiempo
    return wrapper



class Calculadora_de_caminos:
    def __init__(self, N: int, M: int) -> None:
        self.N=N
        self.M=M
        self.grilla = np.zeros((N, M), dtype=int)
    
    @medir_tiempo
    def combinatoria(self):
        caminos = math.comb (self.N + self.M-2, self.M-1)
        return caminos

    @medir_tiempo
    def progra_dinamica(self): 
        for i in range(1,self.N):
            self.grilla[i][0] = 1
        for j in range(1,self.M):
            self.grilla[0][j] = 1
        for i in range(1,self.N):
            for j in range(1,self.M):
                self.grilla[i][j] = self.grilla[i-1][j] + self.grilla[i][j-1]
        return self.grilla[self.N-1][self.M-1]

    @medir_tiempo
    def memoizacion(self,x=0,y=0,memo=None):
        if memo is None:
            memo = {}
        if (x,y) in memo:
            return memo[(x,y)]
        if x == self.N-1 and y == self.M-1:
            return 1
        caminos = 0
        if x+1 < self.N:
            caminos +=self.memoizacion(x+1,y,memo)
        if y+1<self.M:
            caminos +=self.memoizacion(x,y+1,memo)
        memo[(x,y)] = caminos 
        return caminos


sizes = range(2, 12)  # Probar tamaños de grillas desde 2x2 hasta 11x11

# Calcular los tiempos de ejecución para cada tamaño de grilla
for size in sizes:
    calculadora = Calculadora_de_caminos(N=size, M=size)
    
    #Tiempo para combinatoria
    tiempo_combinatoria = calculadora.combinatoria()
    tiempos_combinatoria.append(tiempo_combinatoria)

    #Tiempo para programación dinámica
    tiempo_programacion_dinamica = calculadora.progra_dinamica()
    tiempos_programacion_dinamica.append(tiempo_programacion_dinamica)

    #Tiempo para memoizacion
    tiempo_memoizacion = calculadora.memoizacion()
    tiempos_memoizacion.append(tiempo_memoizacion)


# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(sizes, tiempos_combinatoria, label='Combinatoria (O(1))', marker='x', )
plt.plot(sizes, tiempos_programacion_dinamica, label='Proga. Dinámica (O(N*M))', marker='o')
plt.plot(sizes, tiempos_memoizacion, label='Memoización (O(2^(N+M)))', marker='^')

plt.xlabel('Tamaño de la grilla (N x N)')
plt.ylabel('Tiempo de ejecución (seg)')
plt.title('Comparación de tiempos de ejecución')
plt.legend()
plt.grid(True)
plt.show()




