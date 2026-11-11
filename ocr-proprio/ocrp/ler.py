"""Pipeline completo: documento -> texto + campos validados."""
from __future__ import annotations

import os

from . import campos, documento, preprocess, segment


def ler_imagem(img, reconhecedor, conf_minima=0.80):
    limpo, binaria, info = preprocess.preparar(img)
    blocos = segment.encontrar_blocos(binaria)
    recortes = [limpo[b.y:b.y + b.h, b.x:b.x + b.w] for b in blocos]
    leituras = reconhecedor.ler(recortes) if recortes else []
    linhas_txt, duvidosos = [], []
    i = 0
    for linha in segment.agrupar_linhas(blocos):
        partes = []
        ult_x1 = None
        for b in linha:
            txt, conf, _ = leituras[i]
            if conf < conf_minima:
                duvidosos.append({"texto": txt, "confianca": round(conf, 3), "caixa": b.caixa})
            if ult_x1 is not None:
                partes.append("    " if b.x - ult_x1 > b.h * 1.5 else " ")
            partes.append(txt)
            ult_x1 = b.x + b.w
            i += 1
        linhas_txt.append("".join(partes))
    return "\n".join(linhas_txt), duvidosos, info


def ler_documento(caminho: str, modelo: str = "modelos/ocr_proprio.pt", forcar_ocr: bool = False) -> dict:
    res: dict = {"arquivo": os.path.basename(caminho)}
    if caminho.lower().endswith(".pdf") and not forcar_ocr:
        nativo = documento.texto_nativo_pdf(caminho)
        if nativo:
            res["metodo"] = "texto nativo do PDF (sem OCR, exato)"
            res["texto"] = "\n\n".join(nativo)
    if "texto" not in res:
        from .model import Reconhecedor
        if not os.path.exists(modelo):
            raise SystemExit(f"Modelo {modelo} não encontrado. Rode primeiro: python -m ocrp treinar")
        rec = Reconhecedor(modelo)
        textos, duv, infos = [], [], []
        for pg in documento.paginas_como_imagem(caminho):
            t, d, info = ler_imagem(pg, rec)
            textos.append(t)
            duv += d
            infos.append(info)
        res["metodo"] = "OCR próprio"
        res["texto"] = "\n\n".join(textos)
        res["trechos_duvidosos"] = duv
        res["preprocessamento"] = infos
    res.update(campos.extrair(res["texto"]))
    lanc = campos.lancamentos_extrato(res["texto"])
    if lanc:
        res["lancamentos"] = lanc
        alertas_saldo = campos.conferir_saldos(lanc)
        res["alertas"] += alertas_saldo
        res["confiavel"] = res["confiavel"] and not alertas_saldo
    return res
