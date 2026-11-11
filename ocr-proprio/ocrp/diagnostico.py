"""Responde: POR QUE o OCR errou?

Duas partes:
 1. Qualidade da imagem (resolução, foco, inclinação, contraste, fundo) —
    explica erros que NENHUM motor resolve sem pré-processamento.
 2. Comparação contra o gabarito (dados rotulados): taxa de erro de cada
    motor (Tesseract cru, Tesseract com pré-processamento, OCR próprio) e
    as trocas de caracteres mais comuns (0↔O, 1↔l, 5↔S, ','↔'.' ...).
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import documento, preprocess
from .metrics import cer, confusoes

EXPLICACAO_CONFUSAO = {
    frozenset("0O"): "zero vs letra O — fontes de conta usam formatos quase idênticos",
    frozenset("1l"): "um vs L minúsculo — idem, principalmente em fonte sem serifa",
    frozenset("1I"): "um vs i maiúsculo",
    frozenset("5S"): "5 vs S — típico de impressão fraca/baixa resolução",
    frozenset("8B"): "8 vs B",
    frozenset(",."): "vírgula vs ponto — destrói valores em R$; causado por resolução baixa (o 'rabinho' da vírgula some)",
    frozenset("6G"): "6 vs G",
    frozenset("2Z"): "2 vs Z",
}


def qualidade(img: np.ndarray) -> dict:
    cinza = preprocess.para_cinza(img)
    bin_ = preprocess.binarizar(cinza)
    hx = preprocess.altura_x_estimada(bin_)
    foco = float(cv2.Laplacian(cinza, cv2.CV_64F).var())
    limiar, _ = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tinta, papel = cinza[cinza <= limiar], cinza[cinza > limiar]
    contraste = float(np.median(papel) - np.median(tinta)) if tinta.size and papel.size else 0.0
    ang = preprocess.angulo_inclinacao(bin_)
    blur_bg = cv2.GaussianBlur(cinza, (0, 0), 50)
    iluminacao = float(blur_bg.max() - blur_bg.min())
    problemas = []
    if hx and hx < 12:
        problemas.append(f"Letras com só ~{hx:.0f}px de altura: resolução baixa demais (ideal ≥ 20px, ou escanear a 300 DPI). Causa troca de vírgula/ponto, 5/S, 8/B.")
    if foco < 100:
        problemas.append(f"Imagem desfocada (nitidez {foco:.0f}; abaixo de 100 é borrado). Letras se fundem.")
    if contraste < 80:
        problemas.append(f"Contraste baixo ({contraste:.0f}/255): tinta fraca ou foto escura.")
    if abs(ang) >= 0.5:
        problemas.append(f"Documento inclinado {abs(ang):.1f}°: o OCR lê pedaços de linhas diferentes juntos.")
    if iluminacao > 60:
        problemas.append("Iluminação desigual/sombra (foto de celular): parte do texto some na binarização.")
    if img.ndim == 3 and float(np.std(img.astype(np.float32), axis=2).mean()) > 12:
        problemas.append("Fundo colorido/marca d'água (comum em conta de luz): confunde a separação texto/fundo.")
    return {"altura_letra_px": round(hx, 1), "nitidez": round(foco, 1), "contraste": round(contraste, 1),
            "inclinacao_graus": ang, "variacao_iluminacao": round(iluminacao, 1), "problemas": problemas}


def diagnosticar_arquivo(caminho: str) -> dict:
    out = {"arquivo": caminho}
    if caminho.lower().endswith(".pdf"):
        nativo = documento.texto_nativo_pdf(caminho)
        if nativo:
            out["pdf_tem_texto"] = True
            out["aviso"] = ("Este PDF JÁ TEM TEXTO embutido. Fazer OCR nele é desnecessário e só introduz erro: "
                            "o texto exato pode ser copiado direto (o comando 'ler' faz isso automaticamente).")
    out["paginas"] = [qualidade(p) for p in documento.paginas_como_imagem(caminho)]
    return out


def comparar_motores(pasta_rotulado: str, modelo: str | None = None, limite: int = 300) -> dict:
    from .train import carregar_rotulados
    pares = carregar_rotulados(pasta_rotulado)[:limite]
    if not pares:
        return {"erro": f"Nenhum recorte rotulado em {pasta_rotulado}"}
    imgs = [cv2.imread(c, cv2.IMREAD_GRAYSCALE) for c, _ in pares]
    gab = [t for _, t in pares]
    motores = {}
    try:
        import pytesseract

        def tess(im):
            return pytesseract.image_to_string(im, lang="por", config="--psm 7").strip()

        motores["tesseract (cru)"] = [tess(im) for im in imgs]
        motores["tesseract + pré-proc"] = [tess(_preproc_recorte(im)) for im in imgs]
    except Exception as e:  # noqa: BLE001
        motores["tesseract"] = None
        print(f"(Tesseract indisponível: {e})")
    if modelo and os.path.exists(modelo):
        from .model import Reconhecedor
        motores["OCR próprio"] = [t for t, _, _ in Reconhecedor(modelo).ler(imgs)]
    rel = {}
    for nome, prev in motores.items():
        if prev is None:
            continue
        cs = [cer(p, t) for p, t in zip(prev, gab)]
        conf = confusoes(list(zip(prev, gab))).most_common(12)
        rel[nome] = {
            "CER_%": round(100 * float(np.mean(cs)), 2),
            "linhas_100%_certas_%": round(100 * float(np.mean([p == t for p, t in zip(prev, gab)])), 1),
            "trocas_mais_comuns": [{"certo": a, "leu": b, "vezes": n, "porque": EXPLICACAO_CONFUSAO.get(frozenset(a + b), "")} for (a, b), n in conf],
            "piores_exemplos": [{"certo": t, "leu": p} for p, t, _ in sorted(zip(prev, gab, cs), key=lambda x: -x[2])[:5]],
        }
    return rel


def _preproc_recorte(im):
    h = im.shape[0]
    if h < 40:
        im = cv2.resize(im, None, fx=40 / h, fy=40 / h, interpolation=cv2.INTER_CUBIC)
    return preprocess.binarizar(preprocess.remover_fundo(im))
