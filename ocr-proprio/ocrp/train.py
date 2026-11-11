"""Treina o OCR próprio.

Mistura duas fontes de dados:
  1. sintético, renderizado com as fontes de `fontes/` (ilimitado);
  2. recortes REAIS dos seus documentos, corrigidos por você em
     `dados/rotulado/` (arquivo.png + arquivo.gt.txt). Esses têm peso maior:
     são eles que garantem que o modelo acerte no mundo real.
"""
from __future__ import annotations

import glob
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from . import synth
from .charset import NUM_CLASSES, codificar, normalizar
from .metrics import cer
from .model import ALTURA, CRNN, decodificar, preparar_recorte


def carregar_rotulados(pasta: str) -> list[tuple[str, str]]:
    pares = []
    for gt in sorted(glob.glob(os.path.join(pasta, "**", "*.gt.txt"), recursive=True)):
        base = gt[: -len(".gt.txt")]
        for ext in (".png", ".jpg", ".jpeg", ".tif"):
            if os.path.exists(base + ext):
                txt = normalizar(open(gt, encoding="utf-8").read().strip())
                if txt:
                    pares.append((base + ext, txt))
                break
    return pares


class Misto(IterableDataset):
    def __init__(self, fontes, reais, prop_real=0.5, extra=None, largura_max=480):
        self.fontes, self.reais, self.prop_real, self.extra = fontes, reais, prop_real, extra
        self.largura_max = largura_max

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        random.seed((info.id if info else 0) + int(time.time() * 1000) % 100000)
        np.random.seed(random.randint(0, 2**31 - 1))
        while True:
            if self.reais and (not self.fontes or random.random() < self.prop_real):
                cam, txt = random.choice(self.reais)
                img = cv2.imread(cam, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if random.random() < 0.5:
                    img = synth._augmentar(img)
            else:
                img, txt = synth.amostra(self.fontes, self.extra)
            alvo = codificar(txt)
            if not alvo:
                continue
            x = preparar_recorte(img, largura_max=self.largura_max)
            if x.shape[1] // 4 < len(alvo) * 1.2:  # imagem estreita demais para o texto
                continue
            yield x, alvo, txt


class LotesPorLargura(IterableDataset):
    """Junta exemplos de largura parecida no mesmo lote: menos preenchimento vazio = treino ~2x mais rápido."""

    def __init__(self, fonte: IterableDataset, lote: int, buffer: int = 8):
        self.fonte, self.lote, self.buffer = fonte, lote, buffer

    def __iter__(self):
        it = iter(self.fonte)
        while True:
            itens = sorted((next(it) for _ in range(self.lote * self.buffer)), key=lambda e: e[0].shape[1])
            lotes = [itens[i:i + self.lote] for i in range(0, len(itens), self.lote)]
            random.shuffle(lotes)
            for l in lotes:
                yield colar(l)


class Fixo(Dataset):
    def __init__(self, itens):
        self.itens = itens

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, i):
        return self.itens[i]


def colar(lote):
    imgs, alvos, txts = zip(*lote)
    wmax = max(i.shape[1] for i in imgs)
    x = np.ones((len(imgs), 1, ALTURA, wmax), np.float32)
    for j, im in enumerate(imgs):
        x[j, 0, :, : im.shape[1]] = im
    alvos_flat = torch.tensor([c for a in alvos for c in a], dtype=torch.long)
    tam = torch.tensor([len(a) for a in alvos], dtype=torch.long)
    return torch.from_numpy(x), alvos_flat, tam, list(txts)


def avaliar(modelo, loader, disp):
    modelo.eval()
    soma, n, exatos = 0.0, 0, 0
    with torch.no_grad():
        for x, _, _, txts in loader:
            preds = decodificar(modelo(x.to(disp)).cpu())
            for (p, _, _), t in zip(preds, txts):
                soma += cer(p, t)
                exatos += int(p == t)
                n += 1
    modelo.train()
    return soma / max(1, n), exatos / max(1, n)


def treinar(pasta_fontes="fontes", pasta_rotulado="dados/rotulado", saida="modelos/ocr_proprio.pt",
            passos=20000, lote=32, lr=1e-3, continuar=None, prop_real=0.5, frases_extra=None,
            workers=None, log=print, avaliar_cada=250):
    fontes = synth.listar_fontes(pasta_fontes)
    reais = carregar_rotulados(pasta_rotulado)
    extra = None
    if frases_extra and os.path.exists(frases_extra):
        extra = [normalizar(l.strip()) for l in open(frases_extra, encoding="utf-8") if l.strip()]
    if not fontes and not reais:
        raise SystemExit("Sem dados: coloque fontes .ttf/.otf em 'fontes/' e/ou recortes rotulados em 'dados/rotulado/'.")
    log(f"Fontes: {len(fontes)} | recortes reais rotulados: {len(reais)}")

    random.seed(0)
    # validação: 20% dos reais (se houver) + sintético fixo
    reais_val = random.sample(reais, k=len(reais) // 5) if len(reais) >= 10 else []
    reais_tr = [r for r in reais if r not in reais_val] or reais
    val = []
    if fontes:
        g = iter(Misto(fontes, [], 0, extra))
        val += [next(g) for _ in range(300)]
    for cam, txt in reais_val:
        img = cv2.imread(cam, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            val.append((preparar_recorte(img, 800), codificar(txt), txt))
    val_loader = DataLoader(Fixo(val), batch_size=64, collate_fn=colar)

    disp = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = CRNN().to(disp)
    if continuar and os.path.exists(continuar):
        modelo.load_state_dict(torch.load(continuar, map_location=disp, weights_only=False)["estado"])
        log(f"Continuando de {continuar}")
    otim = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(otim, max_lr=lr, total_steps=passos, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    workers = workers if workers is not None else max(0, min(2, (os.cpu_count() or 1) // 2))
    loader = DataLoader(LotesPorLargura(Misto(fontes, reais_tr, prop_real, extra), lote), batch_size=None, num_workers=workers)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - workers))

    melhor = float("inf")
    os.makedirs(os.path.dirname(saida) or ".", exist_ok=True)
    t0 = time.time()
    perda_media = 0.0
    for passo, (x, alvos, tam, _) in enumerate(loader, 1):
        x = x.to(disp)
        logits = modelo(x)  # B,T,C
        lp = logits.log_softmax(-1).permute(1, 0, 2)
        tam_in = torch.full((x.shape[0],), lp.shape[0], dtype=torch.long)
        perda = ctc(lp, alvos, tam_in, tam)
        otim.zero_grad()
        perda.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 5)
        otim.step()
        sched.step()
        perda_media = 0.98 * perda_media + 0.02 * perda.item() if passo > 1 else perda.item()
        if passo % avaliar_cada == 0 or passo == passos:
            c, ex = avaliar(modelo, val_loader, disp)
            log(f"passo {passo}/{passos}  perda {perda_media:.3f}  CER {c*100:.2f}%  linhas exatas {ex*100:.1f}%  ({time.time()-t0:.0f}s)")
            if c <= melhor:
                melhor = c
                torch.save({"estado": modelo.state_dict(), "oculto": 128, "canais": (32, 64, 96, 96, 128, 128), "cer": c, "passo": passo}, saida)
        if passo >= passos:
            break
    log(f"Melhor CER na validação: {melhor*100:.2f}%  -> {saida}")
    return melhor
