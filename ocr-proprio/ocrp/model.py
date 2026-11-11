"""Rede neural de reconhecimento de linha (CRNN + CTC).

Entrada: recorte de uma linha/bloco em tons de cinza, altura fixa 32 px.
Saída: sequência de caracteres. É a mesma família de arquitetura usada
pelo Tesseract 4+/EasyOCR, mas pequena e treinada só nas SUAS fontes —
por isso acerta mais nos seus documentos e roda rápido em CPU.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from .charset import IDX2CHAR, NUM_CLASSES

ALTURA = 32


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, oculto: int = 128, canais=(32, 64, 96, 96, 128, 128)):
        super().__init__()
        c1, c2, c3, c4, c5, c6 = canais

        def bloco(i, o, pool):
            camadas = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
            if pool:
                camadas.append(nn.MaxPool2d(pool))
            return camadas

        self.cnn = nn.Sequential(
            *bloco(1, c1, (2, 2)),      # 16 x W/2
            *bloco(c1, c2, (2, 2)),     # 8  x W/4
            *bloco(c2, c3, None),
            *bloco(c3, c4, (2, 1)),     # 4  x W/4
            *bloco(c4, c5, None),
            *bloco(c5, c6, (2, 1)),     # 2  x W/4
            nn.Dropout2d(0.1),
        )
        self.rnn = nn.LSTM(c6 * 2, oculto, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(oculto * 2, num_classes)

    def forward(self, x):  # x: B,1,32,W
        f = self.cnn(x)                    # B,C,2,T
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)
        f, _ = self.rnn(f)
        return self.fc(f)                  # B,T,classes


def preparar_recorte(cinza: np.ndarray, largura_max: int = 1600) -> np.ndarray:
    """Normaliza um recorte para a entrada da rede (float32 0..1, texto escuro)."""
    if cinza.ndim == 3:
        cinza = cv2.cvtColor(cinza, cv2.COLOR_BGR2GRAY)
    h, w = cinza.shape
    nw = max(8, min(largura_max, int(round(w * ALTURA / max(1, h)))))
    img = cv2.resize(cinza, (nw, ALTURA), interpolation=cv2.INTER_AREA if h > ALTURA else cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    if hi - lo > 0.05:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    return img


def decodificar(logits: torch.Tensor) -> list[tuple[str, float, list[float]]]:
    """CTC guloso. Devolve (texto, confiança_media, confiança_por_caractere)."""
    probs = logits.softmax(-1)
    conf, idx = probs.max(-1)
    saida = []
    for b in range(idx.shape[0]):
        anterior, chars, confs = 0, [], []
        for t in range(idx.shape[1]):
            k = int(idx[b, t])
            if k != 0 and k != anterior:
                chars.append(IDX2CHAR.get(k, ""))
                confs.append(float(conf[b, t]))
            anterior = k
        media = float(np.mean(confs)) if confs else 0.0
        saida.append(("".join(chars), media, confs))
    return saida


class Reconhecedor:
    def __init__(self, caminho_modelo: str, dispositivo: str | None = None):
        self.disp = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(caminho_modelo, map_location=self.disp, weights_only=False)
        self.modelo = CRNN(oculto=ck.get("oculto", 128), canais=tuple(ck.get("canais", (32, 64, 96, 96, 128, 128)))).to(self.disp)
        self.modelo.load_state_dict(ck["estado"])
        self.modelo.eval()

    @torch.no_grad()
    def ler(self, recortes: list[np.ndarray], lote: int = 32) -> list[tuple[str, float, list[float]]]:
        res = []
        for i in range(0, len(recortes), lote):
            imgs = [preparar_recorte(r) for r in recortes[i:i + lote]]
            wmax = max(im.shape[1] for im in imgs)
            batch = np.ones((len(imgs), 1, ALTURA, wmax), np.float32)
            for j, im in enumerate(imgs):
                batch[j, 0, :, : im.shape[1]] = im
            logits = self.modelo(torch.from_numpy(batch).to(self.disp))
            res.extend(decodificar(logits.cpu()))
        return res
