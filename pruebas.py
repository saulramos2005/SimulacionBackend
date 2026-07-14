import numpy as np
from scipy.stats import chi2, norm, t as t_dist

def _to_native(val):
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

def prueba_K_Smirnov(datos, a=0.0, b=1.0, alpha=0.05):
    datos = np.sort(datos)
    n = len(datos)

    f_teorica = (datos - a) / (b - a)

    f_empirica = np.arange(1, n + 1) / n # i/n
    f_empirica_ant = np.arange(0, n) / n # (i-1)/n

    d_pos = f_empirica - f_teorica # i/n - F0(x_i)
    d_neg = f_teorica - f_empirica_ant # F0(x_i) - (i-1)/n
    
    max_d_pos = np.max(d_pos)
    max_d_neg = np.max(d_neg)
    d_estadistico = max(max_d_pos, max_d_neg)

    # ~1.36/sqrt(n) reescrito
    d_critico = np.sqrt(-0.5 * np.log(alpha / 2)) / np.sqrt(n)

    resultado = {
        "estadistico_D": _to_native(round(d_estadistico, 6)),
        "valor_critico": _to_native(round(d_critico, 6)),
        "rechazar_H0": bool(d_estadistico > d_critico),
        "interpretacion": (
            f"Se rechaza H0: los datos NO provienen de una distribución uniforme ({a}, {b})"
            if d_estadistico > d_critico
            else f"No se rechaza H0: los datos podrían provenir de una distribución uniforme ({a}, {b})"
        ),
        "frecuencia_teorica": _to_native(np.round(f_teorica, 6)),
        "frecuencia_empirica": _to_native(f_empirica),
        "frecuencia_empirica_anterior": _to_native(f_empirica_ant),
        "diferencias_positivas": _to_native(np.round(d_pos, 6)),
        "diferencias_negativas": _to_native(np.round(d_neg, 6))
    }
    return resultado

def prueba_Varianza(datos, sigma_0=np.sqrt(1/12), alpha=0.05):
    datos = np.array(datos)
    n = len(datos)
    s2 = np.var(datos, ddof=1)

    gl = n - 1
    chi2_estadistico = gl * s2 / (sigma_0 ** 2)
    
    # Al ser a dos colas, se calculan ambos límites del intervalo
    chi2_inf = chi2.ppf(alpha / 2, gl)
    chi2_sup = chi2.ppf(1 - alpha / 2, gl)
    
    # Se rechaza si cae fuera del intervalo [chi2_inf, chi2_sup]
    rechazar = bool((chi2_estadistico < chi2_inf) or (chi2_estadistico > chi2_sup))
    critico = (_to_native(round(chi2_inf, 6)), _to_native(round(chi2_sup, 6)))
    p_valor = 2 * min(chi2.cdf(chi2_estadistico, gl), 1 - chi2.cdf(chi2_estadistico, gl))

    resultado = {
        "valor_estadistico": _to_native(round(chi2_estadistico, 6)),
        "grados_libertad": gl,
        "valor_critico": critico,
        "p_valor": _to_native(round(p_valor, 6)),
        "rechazar_H0": rechazar,
        "interpretacion": (
            f"Se rechaza H0: la varianza poblacional difiere significativamente de {round(sigma_0**2, 6)}"
            if rechazar
            else f"No se rechaza H0: la varianza es estadísticamente igual a {round(sigma_0**2, 6)}"
        ),
        "varianza_muestral": _to_native(round(s2, 6)),
        "desviacion_estandar_muestral": _to_native(round(np.sqrt(s2), 6)),
        "varianza_teorica_esperada": _to_native(round(sigma_0 ** 2, 6)),
        "chi2_limite_inferior": _to_native(round(chi2_inf, 6)),
        "chi2_limite_superior": _to_native(round(chi2_sup, 6)),
    }
    return resultado

def prueba_Racha(datos, criterio="mediana", alpha=0.05):
    datos = np.array(datos)
    
    if criterio == "mediana":
        umbral = np.median(datos)
    elif criterio == "media":
        umbral = np.mean(datos)
    elif isinstance(criterio, (int, float)):
        umbral = criterio
    else:
        raise ValueError("criterio debe ser 'mediana', 'media' o un valor numérico")

    # Eliminamos los valores que son exactamente iguales al umbral.
    # En continuas no afectará casi nada; en discretas salvará la simetría de la prueba.
    datos_filtrados = datos[datos != umbral]
    n = len(datos_filtrados)

    if n == 0:
        raise ValueError("La secuencia no tiene variación suficiente para realizar la prueba de rachas.")

    secuencia_b = (datos_filtrados > umbral).astype(int)

    n1 = np.sum(secuencia_b == 1)
    n0 = np.sum(secuencia_b == 0)

    if n1 == 0 or n0 == 0:
        raise ValueError("La secuencia binarizada no tiene variación; no se puede calcular la prueba.")

    rachas = 1
    for i in range(1, n):
        if secuencia_b[i] != secuencia_b[i - 1]:
            rachas += 1

    mu_r = (2 * n1 * n0) / (n1 + n0) + 1 # Media esperada
    sigma_r = np.sqrt((2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))) # Desviacion estandar esperada 

    z_estadistico = (rachas - mu_r) / sigma_r
    z_crit = norm.ppf(1 - alpha / 2)

    p_valor = 2 * (1 - norm.cdf(abs(z_estadistico)))
    rechazar = bool(abs(z_estadistico) > z_crit)

    resultado = {
        "rachas_observadas": int(rachas),
        "rachas_esperadas": _to_native(round(mu_r, 6)),
        "n1": int(n1),
        "n2": int(n0),
        "estadistico_Z": _to_native(round(z_estadistico, 6)),
        "valor_critico_Z": _to_native(round(z_crit, 6)),
        "p_valor": _to_native(round(p_valor, 6)),
        "rechazar_H0": rechazar,
        "interpretacion": (
            "Se rechaza H0: la secuencia NO es aleatoria (hay patrón o tendencia)"
            if rechazar
            else "No se rechaza H0: la secuencia parece ser aleatoria"
        ),
        "umbral_utilizado": _to_native(round(umbral, 6)),
        "criterio": criterio,
        "secuencia_binaria": _to_native(secuencia_b),
        "desviacion_estandar_R": _to_native(round(sigma_r, 6)),
    }
    return resultado

def prueba_Media(datos, mu_0=0.5, sigma=None, alpha=0.05):
    datos = np.array(datos)
    n = len(datos)
    gl = n - 1
    x_barra = np.mean(datos)

    if sigma is not None:
        desviacion = sigma
        critico = norm.ppf(1 - alpha / 2)
        dist = "Z"
    else:
        desviacion = np.std(datos, ddof=1)
        critico = t_dist.ppf(1 - alpha / 2, gl)
        dist = "t"

    estadistico = (x_barra - mu_0) / (desviacion / np.sqrt(n))
    rechazar = bool(abs(estadistico) > critico)
    tupla = (_to_native(round(-critico, 6)), _to_native(round(critico, 6)))
    p_valor = 2 * (1- (norm.cdf(abs(estadistico)) if sigma is not None else t_dist.cdf(abs(estadistico), gl)))

    resultado = {
        "distribucion": dist,
        "estadistico": _to_native(round(estadistico, 6)),
        "media_muestral": _to_native(round(x_barra, 6)),
        "valor_critico": tupla,
        "p_valor": _to_native(round(p_valor, 6)),
        "rechazar_H0": rechazar,
        "interpretacion": (
            f"Se rechaza H0: la media poblacional difiere significativamente de {mu_0}"
            if rechazar
            else f"No se rechaza H0: la media es estadísticamente igual a {mu_0}"
        ),
        "grados_libertad": gl,
        "desviacion_estandar_muestral": _to_native(round(desviacion, 6)),
        "error_estandar_media": _to_native(round((sigma if sigma is not None else desviacion) / np.sqrt(n), 6)),
        "media_esperada_H0": mu_0
    }
    return resultado