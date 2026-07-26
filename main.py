from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional


from distribuciones import DISTRIBUCIONES, PARAMETROS_TEORICOS
from generadores import *
from pruebas import prueba_Bondad_Ajuste, prueba_Varianza, prueba_Media, prueba_Racha


class SimulacionRequest(BaseModel):
    metodo: Literal["congruencial", "medios_cuadrados"]
    distribucion: Literal[
        "uniforme", "exponencial", "normal", "erlang", 
        "bernoulli", "binomial", "poisson"
    ]
    n: int = Field(100, gt=0)
    parametros: dict
    alpha: float = 0.05

app = FastAPI(
    title="Laboratorio Estadistico API",
    description="Backend para generación de muestras y validación estadística",
    version="1.0.0"
)

origins = [
    "http://localhost:5173",
    "https://simulacion-frontend-sigma.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "healthy", "message": "API de Simulación Estadística activa"}


@app.post("/muestras/generar")
def generar_muestras(request: SimulacionRequest):
    nombre_dist = request.distribucion
    
    if nombre_dist not in DISTRIBUCIONES:
        raise HTTPException(status_code=400, detail="Distribución no soportada")
        
    config_dist = DISTRIBUCIONES[nombre_dist]
    
    u_por_muestra = config_dist["u_req"](request.parametros)
    total_u_requeridos = request.n * u_por_muestra

    us_generados = obtener_numeros_u(request.metodo, total_u_requeridos, request.parametros)
    
    muestras = []
    for i in range(request.n):
        inicio = i * u_por_muestra
        fin = inicio + u_por_muestra
        bloque_us = us_generados[inicio:fin]
        
        muestra_calculada = config_dist["func"](bloque_us, request.parametros)
        muestras.append(muestra_calculada)

    pruebas = ejecutar_pruebas(muestras, nombre_dist, request.parametros, request.alpha)

    return {
        "meta": request.dict(),
        "u": us_generados, 
        "x": muestras,
        "pruebas": pruebas
    }

def ejecutar_pruebas(muestras, distribucion, parametros, alpha=0.05):
    resultados = {}

    if distribucion in PARAMETROS_TEORICOS:
        mu_0, sigma_0 = PARAMETROS_TEORICOS[distribucion](parametros)
    else:
        mu_0 = 0.5 
        sigma_0 = 1 / (12**0.5) 
            
    try:
        resultados["Bondad_Ajuste"] = prueba_Bondad_Ajuste(muestras, distribucion, parametros, alpha=alpha)
    except Exception as e:
        resultados["Bondad_Ajuste"] = {"error": str(e)}
    
    try:
        resultados["Varianza"] = prueba_Varianza(muestras, sigma_0=sigma_0, alpha=alpha)
    except Exception as e:
        resultados["Varianza"] = {"error": str(e)}

    try:
        resultados["Media"] = prueba_Media(muestras, mu_0=mu_0, alpha=alpha)
    except Exception as e:
        resultados["Media"] = {"error": str(e)}

    try:
        resultados["Rachas"] = prueba_Racha(muestras, criterio="mediana", alpha=alpha)
    except Exception as e:
        resultados["Rachas"] = {"error": str(e)}

    return resultados