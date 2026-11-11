"""Cria documentos fictícios (conta de luz + extrato) para testar o fluxo inteiro.

Gera: um PDF "digital" (com texto e fonte embutida) e uma "foto" do mesmo
documento (escaneado, torto, borrado, JPEG) — o caso que dá problema.
"""
from __future__ import annotations

import os
import random

import cv2
import numpy as np

from .campos import mod10


def linha_convenio_energia(valor: float) -> str:
    """Linha digitável de conta de consumo (segmento 3 = energia), com DVs válidos (mod 10)."""
    corpo = "836" + "0" + f"{int(round(valor * 100)):011d}" + "0123" + "".join(str(random.randint(0, 9)) for _ in range(25))
    # DV geral (posição 4) em mod10 sobre os outros 43 dígitos
    dv = mod10(corpo[:3] + corpo[4:])
    barras = corpo[:3] + str(dv) + corpo[4:]
    blocos = [barras[i:i + 11] for i in range(0, 44, 11)]
    return " ".join(f"{b}-{mod10(b)}" for b in blocos)


def _fonte_padrao():
    for f in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(f):
            return f
    return None


def gerar_demo(destino="exemplos"):
    import pymupdf
    random.seed(7)
    os.makedirs(destino, exist_ok=True)
    fonte = _fonte_padrao()
    valor = 287.43
    linhas = [
        ("CONCESSIONÁRIA DE ENERGIA EXEMPLO S.A.", 14),
        ("CNPJ: 11.222.333/0001-81", 10),
        ("CONTA DE ENERGIA ELÉTRICA - NOTA FISCAL", 12),
        ("Cliente: MARIA DA SILVA SANTOS", 10),
        ("CPF: 529.982.247-25", 10),
        ("Instalação: 3001234567     Classe: RESIDENCIAL MONOFÁSICO", 10),
        ("Referência: OUT/2026     Vencimento: 25/11/2026", 11),
        ("Leitura anterior: 15/09/2026  48213     Leitura atual: 15/10/2026  48542", 10),
        ("Consumo kWh   329     Bandeira tarifária: AMARELA", 10),
        ("Energia TE   329 kWh   0,312450   102,80", 10),
        ("Energia TUSD   329 kWh   0,402110   132,29", 10),
        ("Contrib. Ilum. Pública   18,34", 10),
        ("ICMS   34,00", 10),
        (f"TOTAL A PAGAR   R$ {valor:.2f}".replace(".", ","), 13),
        (linha_convenio_energia(valor), 11),
    ]
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    if fonte:
        pg.insert_font(fontname="F0", fontfile=fonte)
    y = 60
    for txt, tam in linhas:
        pg.insert_text((40, y), txt, fontsize=tam, fontname="F0" if fonte else "helv")
        y += tam * 2.2
    pdf = os.path.join(destino, "conta_luz_digital.pdf")
    doc.save(pdf)

    # versão "foto/scan ruim" do mesmo documento
    pix = doc[0].get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1), (w, h), borderValue=(235, 235, 235))
    sombra = np.linspace(1.0, 0.7, w)[None, :, None]
    img = np.clip(img.astype(np.float32) * sombra + np.random.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    foto = os.path.join(destino, "conta_luz_foto.jpg")
    cv2.imwrite(foto, img, [cv2.IMWRITE_JPEG_QUALITY, 45])
    gab = os.path.join(destino, "conta_luz_gabarito.txt")
    open(gab, "w", encoding="utf-8").write("\n".join(t for t, _ in linhas) + "\n")
    print("Criados:", pdf, foto, gab)
    return pdf, foto
