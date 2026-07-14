from typing import List
from enum import Enum

class MetodoGeneracion(str, Enum):
    congruencial = "congruencial"
    medios_cuadrados = "medios_cuadrados"

def obtener_numeros_u(metodo: MetodoGeneracion, n: int, parametros: dict = None) -> List[float]:
    parametros = parametros or {}
    if metodo == MetodoGeneracion.congruencial:
        return generarCongruencial(
            seed=parametros.get("seed", 16807),
            mult=parametros.get("mult", 48271),
            mod=parametros.get("mod", 2147483647),
            n=n
        )
    elif metodo == MetodoGeneracion.medios_cuadrados:
        return generarCuadradosMedios(
            seed=parametros.get("seed", 3708),
            d=parametros.get("d", 4),
            n=n
        )
    return []


def generarCongruencial(seed: int = 16807, mult: int = 48271, mod: int = 2147483647, n: int = 100 ):
    cont = 0
    resultados = []
    numero = seed
    while cont < n:
        numero = (numero*mult) % mod
        resultados.append(numero/(mod-1))
        cont += 1
    return resultados

def generarCuadradosMedios(seed: int = 3708, d: int = 4, n: int = 100):
    cont = 0
    resultados = []
    numero = seed

    while cont < n:
        cuadrado = str(numero**2)
        
        # Nos aseguramos de que al menos tenga 'd' dígitos.
        longitud_esperada = max(len(cuadrado), d)
        
        # Si la diferencia no es par, sumamos 1 para poder extraer exactamente del centro
        if (longitud_esperada - d) % 2 != 0:
            longitud_esperada += 1 

        numero_str = cuadrado.zfill(longitud_esperada)

        a = (len(numero_str) - d) // 2
        b = a + d 

        numero = int(numero_str[a:b])
        resultados.append(numero / (10**d))
        cont += 1

    return resultados