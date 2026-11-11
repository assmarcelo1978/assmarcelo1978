"""Pipeline completo: documento -> texto + campos validados."""
from __future__ import annotations

import os

from . import campos, documento, preprocess, segment


def _tesseract_disponivel():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def _ler_tesseract(recortes):
    import cv2
    import pytesseract
    out = []
    for r in recortes:
        h = r.shape[0]
        if h < 40:
            r = cv2.resize(r, None, fx=40 / h, fy=40 / h, interpolation=cv2.INTER_CUBIC)
        r = preprocess.binarizar(r)
        out.append(pytesseract.image_to_string(r, lang="por", config="--psm 7").strip())
    return out


def _comparavel(t):
    return t.replace(" ", "")


def ler_imagem(img, reconhecedor, conf_minima=0.80, segundo_motor=True):
    """Lê a imagem. Com segundo_motor, cada trecho também é lido pelo Tesseract
    (já com o nosso pré-processamento). Se os dois concordam, a leitura é aceita;
    se discordam, o trecho vai para 'trechos_duvidosos' com as duas versões."""
    limpo, binaria, info = preprocess.preparar(img)
    blocos = segment.encontrar_blocos(binaria)
    recortes = [limpo[b.y:b.y + b.h, b.x:b.x + b.w] for b in blocos]
    leituras = reconhecedor.ler(recortes) if recortes else []
    tess = _ler_tesseract(recortes) if (segundo_motor and recortes and _tesseract_disponivel()) else None
    linhas_txt, duvidosos = [], []
    i = 0
    for linha in segment.agrupar_linhas(blocos):
        partes = []
        ult_x1 = None
        for b in linha:
            txt, conf, _ = leituras[i]
            if tess is not None and _comparavel(tess[i]) != _comparavel(txt):
                alternativa = tess[i]
                duvidosos.append({"texto": txt, "tesseract": alternativa, "confianca": round(conf, 3), "caixa": b.caixa})
                if conf < conf_minima and alternativa:
                    txt = alternativa  # modelo próprio inseguro: fica com a outra leitura
            elif conf < conf_minima:
                duvidosos.append({"texto": txt, "confianca": round(conf, 3), "caixa": b.caixa})
            if ult_x1 is not None:
                partes.append("    " if b.x - ult_x1 > b.h * 1.5 else " ")
            partes.append(txt)
            ult_x1 = b.x + b.w
            i += 1
        linhas_txt.append("".join(partes))
    return "\n".join(linhas_txt), duvidosos, info


def ler_documento(caminho: str, modelo: str = "modelos/ocr_proprio.pt", forcar_ocr: bool = False,
                  segundo_motor: bool = True) -> dict:
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
            t, d, info = ler_imagem(pg, rec, segundo_motor=segundo_motor)
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
