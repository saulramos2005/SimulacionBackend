import numpy as np
from scipy.stats import chi2, norm


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


def prueba_K_Smirnov(datos, alpha=0.05):
    datos = np.sort(datos)
    n = len(datos)

    f_teorica = datos

    f_empirica = np.arange(1, n + 1) / n
    f_empirica_ant = np.arange(0, n) / n

    d_pos = np.max(f_empirica - f_teorica)
    d_neg = np.max(f_teorica - f_empirica_ant)
    d_stat = max(d_pos, d_neg)

    d_critico = np.sqrt(-0.5 * np.log(alpha / 2)) / np.sqrt(n)

    resultado = {
        "estadistico_D": _to_native(round(d_stat, 6)),
        "valor_critico": _to_native(round(d_critico, 6)),
        "n": n,
        "alpha": alpha,
        "rechazar_H0": bool(d_stat > d_critico),
        "interpretacion": (
            "Se rechaza H0: los datos NO provienen de una distribución uniforme (0,1)"
            if d_stat > d_critico
            else "No se rechaza H0: los datos podrían provenir de una distribución uniforme (0,1)"
        ),
    }
    return resultado


def prueba_Varianza(datos, sigma_0=np.sqrt(1/12), alpha=0.05, cola="dos"):
    datos = np.array(datos)
    n = len(datos)
    s2 = np.var(datos, ddof=1)

    chi2_stat = (n - 1) * s2 / (sigma_0 ** 2)
    gl = n - 1

    if cola == "dos":
        chi2_inf = chi2.ppf(alpha / 2, gl)
        chi2_sup = chi2.ppf(1 - alpha / 2, gl)
        rechazar = bool((chi2_stat < chi2_inf) or (chi2_stat > chi2_sup))
        critico = (_to_native(round(chi2_inf, 6)), _to_native(round(chi2_sup, 6)))
        p_valor = 2 * min(chi2.cdf(chi2_stat, gl), 1 - chi2.cdf(chi2_stat, gl))
    elif cola == "mayor":
        chi2_crit = chi2.ppf(1 - alpha, gl)
        rechazar = bool(chi2_stat > chi2_crit)
        critico = _to_native(round(chi2_crit, 6))
        p_valor = 1 - chi2.cdf(chi2_stat, gl)
    elif cola == "menor":
        chi2_crit = chi2.ppf(alpha, gl)
        rechazar = bool(chi2_stat < chi2_crit)
        critico = _to_native(round(chi2_crit, 6))
        p_valor = chi2.cdf(chi2_stat, gl)
    else:
        raise ValueError("cola debe ser 'dos', 'mayor' o 'menor'")

    resultado = {
        "valor_estadistico": _to_native(round(chi2_stat, 6)),
        "grados_libertad": gl,
        "valor_critico": critico,
        "p_valor": _to_native(round(p_valor, 6)),
        "lim_sup": chi2_sup | chi2_crit,
        "alpha": alpha,
        "cola": cola,
        "rechazar_H0": rechazar,
        "interpretacion": (
            f"Se rechaza H0: la varianza poblacional es {'diferente' if cola == 'dos' else ('mayor' if cola == 'mayor' else 'menor')} a {round(sigma_0**2, 6)}"
            if rechazar
            else f"No se rechaza H0: no hay evidencia suficiente para afirmar que la varianza {'difiere' if cola == 'dos' else ('sea mayor' if cola == 'mayor' else 'sea menor')} de {round(sigma_0**2, 6)}"
        ),
    }
    return resultado


def prueba_Media(datos, mu_0=0.5, sigma=None, alpha=0.05, cola="dos"):
    datos = np.array(datos)
    n = len(datos)
    x_bar = np.mean(datos)

    if sigma is not None:
        estadistico = (x_bar - mu_0) / (sigma / np.sqrt(n))
        dist = "Z"
        if cola == "dos":
            z_crit = norm.ppf(1 - alpha / 2)
            rechazar = bool(abs(estadistico) > z_crit)
            critico = _to_native(round(z_crit, 6))
        elif cola == "mayor":
            z_crit = norm.ppf(1 - alpha)
            rechazar = bool(estadistico > z_crit)
            critico = _to_native(round(z_crit, 6))
        elif cola == "menor":
            z_crit = norm.ppf(alpha)
            rechazar = bool(estadistico < z_crit)
            critico = _to_native(round(z_crit, 6))
        else:
            raise ValueError("cola debe ser 'dos', 'mayor' o 'menor'")
        if cola == "dos":
            p_valor = 2 * (1 - norm.cdf(abs(estadistico)))
        elif cola == "mayor":
            p_valor = 1 - norm.cdf(estadistico)
        else:
            p_valor = norm.cdf(estadistico)
    else:
        s = np.std(datos, ddof=1)
        estadistico = (x_bar - mu_0) / (s / np.sqrt(n))
        gl = n - 1
        dist = "t"
        if cola == "dos":
            t_crit = 2 * (1 - norm.cdf(abs(estadistico)))
            from scipy.stats import t as t_dist
            t_crit_val = t_dist.ppf(1 - alpha / 2, gl)
            rechazar = bool(abs(estadistico) > t_crit_val)
            critico = _to_native(round(t_crit_val, 6))
            p_valor = t_crit
        elif cola == "mayor":
            from scipy.stats import t as t_dist
            t_crit_val = t_dist.ppf(1 - alpha, gl)
            rechazar = bool(estadistico > t_crit_val)
            critico = _to_native(round(t_crit_val, 6))
            p_valor = 1 - t_dist.cdf(estadistico, gl)
        elif cola == "menor":
            from scipy.stats import t as t_dist
            t_crit_val = t_dist.ppf(alpha, gl)
            rechazar = bool(estadistico < t_crit_val)
            critico = _to_native(round(t_crit_val, 6))
            p_valor = t_dist.cdf(estadistico, gl)
        else:
            raise ValueError("cola debe ser 'dos', 'mayor' o 'menor'")

    resultado = {
        "distribucion": dist,
        "estadistico": _to_native(round(estadistico, 6)),
        "media_muestral": _to_native(round(x_bar, 6)),
        "valor_critico": critico,
        "p_valor": _to_native(round(p_valor, 6)),
        "alpha": alpha,
        "cola": cola,
        "rechazar_H0": rechazar,
        "interpretacion": (
            f"Se rechaza H0: la media poblacional es {'diferente' if cola == 'dos' else ('mayor' if cola == 'mayor' else 'menor')} a {mu_0}"
            if rechazar
            else f"No se rechaza H0: no hay evidencia suficiente para afirmar que la media {'difiere' if cola == 'dos' else ('sea mayor' if cola == 'mayor' else 'sea menor')} de {mu_0}"
        ),
    }
    return resultado


def prueba_Racha(datos, criterio="mediana", alpha=0.05):
    datos = np.array(datos)
    n = len(datos)

    if criterio == "mediana":
        umbral = 0.5
    elif criterio == "media":
        umbral = np.mean(datos)
    elif isinstance(criterio, (int, float)):
        umbral = criterio
    else:
        raise ValueError("criterio debe ser 'mediana', 'media' o un valor numérico")

    secuencia = (datos >= umbral).astype(int)

    n1 = np.sum(secuencia == 1)
    n2 = np.sum(secuencia == 0)

    if n1 == 0 or n2 == 0:
        raise ValueError("La secuencia no tiene variación; no se puede realizar la prueba de rachas.")

    rachas = 1
    for i in range(1, n):
        if secuencia[i] != secuencia[i - 1]:
            rachas += 1

    mu_r = (2 * n1 * n2) / (n1 + n2) + 1
    sigma_r = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1)))

    z_stat = (rachas - mu_r) / sigma_r
    z_crit = norm.ppf(1 - alpha / 2)

    p_valor = 2 * (1 - norm.cdf(abs(z_stat)))
    rechazar = bool(abs(z_stat) > z_crit)

    resultado = {
        "rachas_observadas": int(rachas),
        "rachas_esperadas": _to_native(round(mu_r, 6)),
        "n1": int(n1),
        "n2": int(n2),
        "estadistico_Z": _to_native(round(z_stat, 6)),
        "valor_critico_Z": _to_native(round(z_crit, 6)),
        "p_valor": _to_native(round(p_valor, 6)),
        "alpha": alpha,
        "rechazar_H0": rechazar,
        "interpretacion": (
            "Se rechaza H0: la secuencia NO es aleatoria (hay patrón o tendencia)"
            if rechazar
            else "No se rechaza H0: la secuencia parece ser aleatoria"
        ),
    }
    return resultado
