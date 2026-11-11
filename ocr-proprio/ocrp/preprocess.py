"""Pré-processamento: a maior causa de erro de OCR é imagem ruim, não o motor.

Fluxo: tons de cinza -> corrige escala (DPI) -> corrige inclinação ->
remove fundo/sombra -> binariza de forma adaptativa.
"""
from __future__ import annotations

import cv2
import numpy as np

ALTURA_X_ALVO = 20  # altura típica de uma letra minúscula, em pixels, depois de normalizar


def para_cinza(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def remover_fundo(cinza: np.ndarray) -> np.ndarray:
    """Remove sombras, marca d'água clara e fundo colorido (comum em contas de luz)."""
    k = max(15, (min(cinza.shape) // 40) | 1)
    fundo = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    diff = cv2.divide(cinza, fundo, scale=255)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def binarizar(cinza: np.ndarray) -> np.ndarray:
    """Binarização adaptativa (texto preto=0, papel branco=255)."""
    bloco = max(31, (min(cinza.shape) // 30) | 1)
    return cv2.adaptiveThreshold(cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bloco, 15)


def angulo_inclinacao(binaria: np.ndarray) -> float:
    """Estima a inclinação (graus) pelas linhas de texto. Foto torta = OCR errado."""
    tinta = 255 - binaria
    linhas = cv2.morphologyEx(tinta, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3)))
    pts = cv2.findNonZero(linhas)
    if pts is None or len(pts) < 100:
        return 0.0
    melhor, melhor_score = 0.0, -1.0
    h, w = tinta.shape
    pequena = cv2.resize(tinta, (w // 2, h // 2)) if max(h, w) > 1200 else tinta
    for ang in np.arange(-10, 10.01, 0.25):
        M = cv2.getRotationMatrix2D((pequena.shape[1] // 2, pequena.shape[0] // 2), ang, 1.0)
        rot = cv2.warpAffine(pequena, M, (pequena.shape[1], pequena.shape[0]))
        perfil = rot.sum(axis=1).astype(np.float64)
        score = float(np.var(perfil))
        if score > melhor_score:
            melhor, melhor_score = float(ang), score
    return melhor


def rotacionar(img: np.ndarray, ang: float) -> np.ndarray:
    if abs(ang) < 0.1:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def altura_x_estimada(binaria: np.ndarray) -> float:
    """Altura mediana dos componentes de texto — mede se a resolução é suficiente."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(255 - binaria, 8)
    alturas = [s[cv2.CC_STAT_HEIGHT] for s in stats[1:] if 4 <= s[cv2.CC_STAT_HEIGHT] <= 200 and s[cv2.CC_STAT_AREA] > 8]
    return float(np.median(alturas)) if alturas else 0.0


def preparar(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Retorna (cinza_limpo, binaria, info)."""
    cinza = para_cinza(img)
    info: dict = {}
    b0 = binarizar(cinza)
    hx = altura_x_estimada(b0)
    info["altura_letra_px"] = hx
    if 0 < hx < ALTURA_X_ALVO:
        fator = min(4.0, ALTURA_X_ALVO / hx)
        cinza = cv2.resize(cinza, None, fx=fator, fy=fator, interpolation=cv2.INTER_CUBIC)
        info["ampliado"] = round(fator, 2)
    limpo = remover_fundo(cinza)
    limpo = cv2.fastNlMeansDenoising(limpo, None, 7, 7, 21)
    bin_ = binarizar(limpo)
    ang = angulo_inclinacao(bin_)
    info["inclinacao_graus"] = ang
    if abs(ang) >= 0.1:
        limpo = rotacionar(limpo, ang)
        bin_ = binarizar(limpo)
    return limpo, bin_, info
