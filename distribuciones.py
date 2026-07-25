import math

# --- CONTINUAS ---

def inv_uniforme(u, parametros):
    a, b = float(parametros.get("a", 0.0)), float(parametros.get("b", 1.0))
    return a + u[0] * (b - a)

def inv_exponencial(u, parametros):
    lam = float(parametros.get("lam", 1.0))
    u_seguro = max(u[0], 1e-10)
    return -(1.0 / lam) * math.log(u_seguro)

def conv_normal(u, parametros):
    # Aproximaicon por TLC (requiere 12 'u' por cada 'x')
    mu = float(parametros.get("mu", 0.0))
    sigma = float(parametros.get("sigma", 1.0))
    suma_u = sum(u)
    return mu + sigma * (suma_u - 6.0)

def conv_erlang(u, parametros):
    # Requiere 'k' números 'u' por cada 'x'
    lam = float(parametros.get("lam", 1.0))
    prod_u = max(math.prod(u), 1e-10)
    return -(1.0 / lam) * math.log(prod_u)


# --- DISCRETAS ---

def inv_bernoulli(u, parametros):
    p = float(parametros.get("p", 0.5))
    return 1 if u[0] <= p else 0

def conv_binomial(u, parametros):
    # Requiere 'n_ensayos' números 'u' para cada 'x'
    p = float(parametros.get("p", 0.5))
    return sum(1 for u in u if u <= p)

def inv_poisson(u, parametros):
    lam = float(parametros.get("lam", 1.0))
    x = 0
    p_x = math.exp(-lam)
    F_x = p_x
    while u[0] > F_x:
        x += 1
        p_x = p_x * (lam / x)
        F_x += p_x
    return x

# --- DICCIONARIO DE 'U' REQUERIDAS Y FUNCIONES A EJECUTAR ---
DISTRIBUCIONES = {
    "uniforme":    {"u_req": lambda p: 1,                           "func": inv_uniforme},
    "exponencial": {"u_req": lambda p: 1,                           "func": inv_exponencial},
    "normal":      {"u_req": lambda p: 12,                          "func": conv_normal},
    "erlang":      {"u_req": lambda p: int(p.get("k", 1)),          "func": conv_erlang},
    "bernoulli":   {"u_req": lambda p: 1,                           "func": inv_bernoulli},
    "binomial":    {"u_req": lambda p: int(p.get("n_ensayos", 1)),  "func": conv_binomial},
    "poisson":     {"u_req": lambda p: 1,                           "func": inv_poisson}
}

# --- DICCIONARIO DE ESPERANZAS Y DESVIACIONES TEORICAS ---
PARAMETROS_TEORICOS = {
    "uniforme": lambda p: (
        (float(p.get("a", 0.0)) + float(p.get("b", 1.0))) / 2,
        (float(p.get("b", 1.0)) - float(p.get("a", 0.0))) / (12**0.5)
    ),
    "exponencial": lambda p: (
        1.0 / float(p.get("lam", 1.0)),
        1.0 / float(p.get("lam", 1.0))
    ),
    "normal": lambda p: (
        float(p.get("mu", 0.0)),
        float(p.get("sigma", 1.0))
    ),
    "erlang": lambda p: (
        int(p.get("k", 1)) / float(p.get("lam", 1.0)),
        (int(p.get("k", 1))**0.5) / float(p.get("lam", 1.0))
    ),
    "bernoulli": lambda p: (
        float(p.get("p", 0.5)),
        (float(p.get("p", 0.5)) * (1.0 - float(p.get("p", 0.5))))**0.5
    ),
    "binomial": lambda p: (
        int(p.get("n_ensayos", 1)) * float(p.get("p", 0.5)),
        (int(p.get("n_ensayos", 1)) * float(p.get("p", 0.5)) * (1.0 - float(p.get("p", 0.5))))**0.5
    ),
    "poisson": lambda p: (
        float(p.get("lam", 1.0)),
        float(p.get("lam", 1.0))**0.5
    )
}