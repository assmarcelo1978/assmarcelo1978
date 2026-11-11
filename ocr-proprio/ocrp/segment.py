"""Encontra as linhas de texto e as divide em blocos (colunas).

Extratos e contas são tabelas: "DATA  HISTÓRICO  VALOR". Se o OCR lê a
linha inteira de uma vez ele mistura colunas e "come" espaços. Aqui cada
bloco separado por um espaço grande vira um pedaço independente, com a
posição (x, y) guardada para a extração de campos.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Bloco:
    x: int
    y: int
    w: int
    h: int

    @property
    def caixa(self):
        return (self.x, self.y, self.w, self.h)


def _remover_linhas_tabela(tinta: np.ndarray) -> np.ndarray:
    """Remove as linhas horizontais/verticais das tabelas (viram 'l', '|', '_' no OCR)."""
    h, w = tinta.shape
    horiz = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1)))
    vert = cv2.morphologyEx(tinta, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 25))))
    return cv2.subtract(tinta, cv2.bitwise_or(horiz, vert))


def encontrar_blocos(binaria: np.ndarray, fator_espaco: float = 1.6) -> list[Bloco]:
    tinta = _remover_linhas_tabela(255 - binaria)
    n, _, stats, _ = cv2.connectedComponentsWithStats(tinta, 8)
    comps = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 6 and s[cv2.CC_STAT_HEIGHT] >= 4]
    if not comps:
        return []
    alt_med = float(np.median([c[cv2.CC_STAT_HEIGHT] for c in comps]))
    # junta letras em palavras e palavras próximas em blocos
    kx = max(3, int(alt_med * fator_espaco))
    ky = max(1, int(alt_med * 0.15))
    juntado = cv2.dilate(tinta, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(juntado, 8)
    blocos = []
    H, W = binaria.shape
    for x, y, w, h, area in stats[1:]:
        if h < alt_med * 0.5 or h > alt_med * 6 or w < alt_med * 0.5:
            continue
        # recorta de volta para a tinta real (tira a dilatação)
        sub = tinta[y:y + h, x:x + w]
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            continue
        x0, x1 = x + xs.min(), x + xs.max() + 1
        y0, y1 = y + ys.min(), y + ys.max() + 1
        pad = max(2, int(alt_med * 0.25))
        blocos.append(Bloco(max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad) - max(0, x0 - pad), min(H, y1 + pad) - max(0, y0 - pad)))
    return ordenar_leitura(blocos)


def ordenar_leitura(blocos: list[Bloco]) -> list[Bloco]:
    """Ordem de leitura: de cima para baixo, e da esquerda para a direita na mesma linha."""
    blocos = sorted(blocos, key=lambda b: b.y + b.h / 2)
    linhas: list[list[Bloco]] = []
    for b in blocos:
        cy = b.y + b.h / 2
        if linhas:
            ult = linhas[-1]
            cy_l = np.mean([o.y + o.h / 2 for o in ult])
            h_l = np.median([o.h for o in ult])
            if abs(cy - cy_l) < h_l * 0.5:
                ult.append(b)
                continue
        linhas.append([b])
    out = []
    for ln in linhas:
        out.extend(sorted(ln, key=lambda b: b.x))
    return out


def agrupar_linhas(blocos: list[Bloco]) -> list[list[Bloco]]:
    linhas: list[list[Bloco]] = []
    for b in blocos:
        cy = b.y + b.h / 2
        if linhas:
            ult = linhas[-1]
            cy_l = np.mean([o.y + o.h / 2 for o in ult])
            if abs(cy - cy_l) < np.median([o.h for o in ult]) * 0.5:
                ult.append(b)
                continue
        linhas.append([b])
    return linhas
