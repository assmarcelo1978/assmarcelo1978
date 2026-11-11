"""Gera recortes dos SEUS documentos para você corrigir (vira dado de treino real).

Para cada documento: pré-processa, acha os blocos de texto, salva cada um em
`dados/para_rotular/` como PNG + .gt.txt já preenchido com o palpite do
modelo atual (ou do Tesseract, ou do texto nativo do PDF). Você só abre o
.gt.txt e corrige o que estiver errado. Depois move para `dados/rotulado/`.

Atalho valioso: se o documento é um PDF com texto, o rótulo vem do próprio
PDF — é 100% correto, sem digitar nada. Basta ter o PDF digital de uma
conta igual às que você recebe em foto/scan.
"""
from __future__ import annotations

import os

import cv2

from . import documento, preprocess, segment


def _rotulos_pdf_por_caixa(caminho, dpi=300):
    """Para PDF com texto: rótulo exato de cada palavra/linha, com a caixa na imagem renderizada."""
    import pymupdf
    doc = pymupdf.open(caminho)
    escala = dpi / 72
    paginas = []
    for pg in doc:
        linhas = {}
        for x0, y0, x1, y1, w, b, l, _ in pg.get_text("words"):
            linhas.setdefault((b, l), []).append((x0, y0, x1, y1, w))
        itens = []
        for ws in linhas.values():
            ws.sort()
            # quebra a linha onde houver espaço grande (mesma lógica do segmentador)
            grupo = [ws[0]]
            for w in ws[1:]:
                alt = w[3] - w[1]
                if w[0] - grupo[-1][2] > alt * 1.6:
                    itens.append(grupo)
                    grupo = [w]
                else:
                    grupo.append(w)
            itens.append(grupo)
        caixas = []
        for g in itens:
            x0, y0 = min(w[0] for w in g), min(w[1] for w in g)
            x1, y1 = max(w[2] for w in g), max(w[3] for w in g)
            pad = (y1 - y0) * 0.2
            caixas.append(((int((x0 - pad) * escala), int((y0 - pad) * escala), int((x1 + pad) * escala), int((y1 + pad) * escala)), " ".join(w[4] for w in g)))
        paginas.append(caixas)
    return paginas


def gerar(caminhos: list[str], destino: str = "dados/para_rotular", modelo: str | None = None, usar_tesseract=False) -> int:
    os.makedirs(destino, exist_ok=True)
    rec = None
    if modelo and os.path.exists(modelo):
        from .model import Reconhecedor
        rec = Reconhecedor(modelo)
    n = 0
    for cam in caminhos:
        base = os.path.splitext(os.path.basename(cam))[0]
        pags = documento.paginas_como_imagem(cam)
        if cam.lower().endswith(".pdf") and documento.texto_nativo_pdf(cam):
            for p, (img, caixas) in enumerate(zip(pags, _rotulos_pdf_por_caixa(cam))):
                for k, ((x0, y0, x1, y1), txt) in enumerate(caixas):
                    rec_img = img[max(0, y0):y1, max(0, x0):x1]
                    if rec_img.size == 0:
                        continue
                    nome = os.path.join(destino, f"{base}_p{p}_{k:04d}")
                    cv2.imwrite(nome + ".png", rec_img)
                    open(nome + ".gt.txt", "w", encoding="utf-8").write(txt + "\n")
                    n += 1
            continue
        for p, img in enumerate(pags):
            limpo, binaria, _ = preprocess.preparar(img)
            blocos = segment.encontrar_blocos(binaria)
            recortes = [limpo[b.y:b.y + b.h, b.x:b.x + b.w] for b in blocos]
            if rec:
                palpites = [t for t, _, _ in rec.ler(recortes)]
            elif usar_tesseract:
                import pytesseract
                palpites = [pytesseract.image_to_string(r, lang="por", config="--psm 7").strip() for r in recortes]
            else:
                palpites = [""] * len(recortes)
            for k, (r, t) in enumerate(zip(recortes, palpites)):
                nome = os.path.join(destino, f"{base}_p{p}_{k:04d}")
                cv2.imwrite(nome + ".png", r)
                open(nome + ".gt.txt", "w", encoding="utf-8").write(t + "\n")
                n += 1
    return n
