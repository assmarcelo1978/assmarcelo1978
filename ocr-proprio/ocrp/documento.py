"""Abre o documento e decide o melhor caminho de leitura.

Regra de ouro: se o PDF já tem texto (extrato baixado do internet banking,
fatura digital da concessionária), NÃO faz OCR — copia o texto, que é 100%
exato. OCR só entra para foto, scan ou PDF que é só imagem.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

EXT_IMAGEM = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _abrir_pdf(caminho):
    import pymupdf
    return pymupdf.open(caminho)


def texto_nativo_pdf(caminho: str) -> list[str] | None:
    """Texto embutido do PDF, por página, mantendo a ordem visual. None se for só imagem."""
    doc = _abrir_pdf(caminho)
    paginas = []
    total = 0
    for pg in doc:
        palavras = pg.get_text("words")  # x0,y0,x1,y1,palavra,bloco,linha,n
        total += len(palavras)
        linhas: list[list] = []
        for w in sorted(palavras, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
            cy = (w[1] + w[3]) / 2
            if linhas and abs(cy - linhas[-1][0]) < (w[3] - w[1]) * 0.5:
                linhas[-1][1].append(w)
            else:
                linhas.append([cy, [w]])
        txt = []
        for _, ws in linhas:
            ws.sort(key=lambda w: w[0])
            partes, ult_x1 = [], None
            for w in ws:
                if ult_x1 is not None:
                    gap = w[0] - ult_x1
                    partes.append("    " if gap > (w[3] - w[1]) * 1.5 else " ")
                partes.append(w[4])
                ult_x1 = w[2]
            txt.append("".join(partes))
        paginas.append("\n".join(txt))
    return paginas if total >= 20 else None


def paginas_como_imagem(caminho: str, dpi: int = 300) -> list[np.ndarray]:
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".pdf":
        import pymupdf
        doc = _abrir_pdf(caminho)
        imgs = []
        for pg in doc:
            pix = pg.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            imgs.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy())
        return imgs
    img = cv2.imdecode(np.fromfile(caminho, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não consegui abrir a imagem: {caminho}")
    return [img]


def extrair_fontes_pdf(caminho: str, pasta_destino: str) -> list[str]:
    """Salva as fontes embutidas no PDF (as MESMAS usadas na conta/extrato).

    É a forma mais fiel de 'treinar com as fontes dos documentos'.
    """
    doc = _abrir_pdf(caminho)
    os.makedirs(pasta_destino, exist_ok=True)
    salvas, vistas = [], set()
    for pg in doc:
        for f in pg.get_fonts(full=True):
            xref, ext, _tipo, nome = f[0], f[1], f[2], f[3]
            if xref in vistas or ext in ("n/a", ""):
                continue
            vistas.add(xref)
            nome_f, ext_f, _, conteudo = doc.extract_font(xref)
            if not conteudo:
                continue
            ext_f = {"ttf": "ttf", "otf": "otf", "cff": "otf", "pfa": "pfa", "pfb": "pfb"}.get(ext_f, ext_f)
            limpo = "".join(c for c in (nome_f or nome) if c.isalnum() or c in "-_+")
            destino = os.path.join(pasta_destino, f"{limpo}.{ext_f}")
            with open(destino, "wb") as fh:
                fh.write(conteudo)
            salvas.append(destino)
    return salvas
