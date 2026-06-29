from pydantic import BaseModel

class ParametrosUniforme(BaseModel):
    a: float = 0.0
    b: float = 1.0

def inversa_uniforme(u: float, a: float, b: float):
    return a + u*(b - a)

# Ej. Para cuando se implementen las otras:
class ParametrosExponencial(BaseModel):
    lambd: float = 1.0