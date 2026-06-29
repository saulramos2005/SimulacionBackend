from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional


from distribuciones import inversa_uniforme
from generadores import *
from pruebas import prueba_K_Smirnov, prueba_Varianza, prueba_Media, prueba_Racha


class SimulacionRequest(BaseModel):
    metodo: Literal["congruencial", "medios_cuadrados"]
    distribucion: Literal["uniforme", "exponencial"]
    n: int = Field(100, gt=0)
    parametros: dict
    a: Optional[float] = None
    b: Optional[float] = None
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

    u = obtener_numeros_u(request.metodo, request.n, request.parametros)

    if request.distribucion == "uniforme":
        a = request.a if request.a is not None else request.parametros.get("a", 0.0)
        b = request.b if request.b is not None else request.parametros.get("b", 1.0)
        muestras = [inversa_uniforme(val, a, b) for val in u]
    else:
        raise HTTPException(status_code=400, detail="Distribución no soportada")

    pruebas = ejecutar_pruebas(muestras, request.distribucion, request.alpha)

    return {
        "meta": request.dict(),
        "data": muestras,
        "pruebas": pruebas
    }


def ejecutar_pruebas(muestras, distribucion, alpha=0.05):
    resultados = {}

    try:
        resultados["Kolmogorov_Smirnov"] = prueba_K_Smirnov(muestras, alpha=alpha)
    except Exception as e:
        resultados["Kolmogorov_Smirnov"] = {"error": str(e)}

    if distribucion == "uniforme":
        sigma_0 = 1 / (2 * (3 ** 0.5))
        mu_0 = 0.5

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