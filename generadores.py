from typing import List
from enum import Enum

class MetodoGeneracion(str, Enum):
    congruencial = "congruencial"
    medios_cuadrados = "medios_cuadrados"

def obtener_numeros_u(metodo: MetodoGeneracion, n: int, parametros: dict = None) -> List[float]:
    parametros = parametros or {}
    if metodo == MetodoGeneracion.congruencial:
        return generarCM(
            seed=parametros.get("semilla", 16807),
            mult=parametros.get("mult", 48271),
            mod=parametros.get("mod", 2147483647),
            n=n
        )
    elif metodo == MetodoGeneracion.medios_cuadrados:
        return generarMC(
            seed=parametros.get("semilla", 3708),
            d=parametros.get("d", 4),
            n=n
        )
    return []


def generarCM(seed: int = 16807, mult: int = 48271, mod: int = 2147483647, n: int = 100 ):
    cont = 0
    resultados = []
    numero = seed
    while cont < n:
        numero = (numero*mult) % mod
        resultados.append(numero/mod)
        cont += 1
    return resultados

def generarMC(seed: int = 3708, d: int = 4, n: int = 100 ):
    cont = 0
    resultados = []
    numero = seed
    while cont < n:
        cuadrado = str(numero**2)
        longitud = max(len(cuadrado) + 1, 2*d) if (d % 2 == 0) else max(len(cuadrado), 2*d + 1)  
        numero = cuadrado.zfill(longitud)
        a = ((len(numero) - d) // 2)
        b = a + d 
        numero = int(numero[a:b])
        resultados.append(numero/(10**d))
        cont += 1
    return resultados

