"""Métricas de erro e análise de confusões."""
from __future__ import annotations

from collections import Counter


def alinhar(a: str, b: str):
    """Alinhamento de Levenshtein. Retorna lista de (op, char_previsto, char_certo)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("=" if a[i - 1] == b[j - 1] else "S", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("I", a[i - 1], ""))  # inseriu caractere que não existe
            i -= 1
        else:
            ops.append(("D", "", b[j - 1]))  # deixou de ler um caractere
            j -= 1
    return ops[::-1]


def distancia(a: str, b: str) -> int:
    return sum(op != "=" for op, _, _ in alinhar(a, b))


def cer(previsto: str, certo: str) -> float:
    """Character Error Rate: % de caracteres errados."""
    if not certo:
        return float(bool(previsto))
    return distancia(previsto, certo) / len(certo)


def confusoes(pares: list[tuple[str, str]]) -> Counter:
    c: Counter = Counter()
    for p, t in pares:
        for op, cp, ct in alinhar(p, t):
            if op == "S":
                c[(ct, cp)] += 1
            elif op == "D":
                c[(ct, "∅")] += 1
            elif op == "I":
                c[("∅", cp)] += 1
    return c
