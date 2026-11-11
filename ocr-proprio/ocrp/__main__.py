"""Linha de comando:  python -m ocrp <comando> ...

  extrair-fontes  PDF...            copia as fontes embutidas nos PDFs para fontes/
  treinar                           treina o OCR próprio (fontes/ + dados/rotulado/)
  rotular         ARQ...            gera recortes dos seus documentos para corrigir
  ler             ARQ...            lê documento e extrai campos validados (JSON)
  diagnosticar    ARQ...            explica a qualidade da imagem / por que erra
  comparar                          compara Tesseract x OCR próprio no gabarito
  demo                              cria documentos de exemplo para testar
"""
from __future__ import annotations

import argparse
import json
import sys

MODELO_PADRAO = "modelos/ocr_proprio.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ocrp", description="OCR próprio para contas, extratos e boletos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("extrair-fontes")
    s.add_argument("pdfs", nargs="+")
    s.add_argument("--destino", default="fontes")

    s = sub.add_parser("treinar")
    s.add_argument("--fontes", default="fontes")
    s.add_argument("--rotulado", default="dados/rotulado")
    s.add_argument("--saida", default=MODELO_PADRAO)
    s.add_argument("--passos", type=int, default=20000)
    s.add_argument("--lote", type=int, default=32)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--continuar", default=None, help="modelo .pt para continuar treinando (ajuste fino)")
    s.add_argument("--prop-real", type=float, default=0.5, help="fração de exemplos reais em cada lote")
    s.add_argument("--frases", default=None, help="arquivo .txt com frases reais dos seus documentos (uma por linha)")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--avaliar-cada", type=int, default=250)

    s = sub.add_parser("rotular")
    s.add_argument("arquivos", nargs="+")
    s.add_argument("--destino", default="dados/para_rotular")
    s.add_argument("--modelo", default=MODELO_PADRAO)
    s.add_argument("--tesseract", action="store_true", help="pré-preenche com o Tesseract se não houver modelo")

    s = sub.add_parser("ler")
    s.add_argument("arquivos", nargs="+")
    s.add_argument("--modelo", default=MODELO_PADRAO)
    s.add_argument("--forcar-ocr", action="store_true", help="faz OCR mesmo em PDF com texto")
    s.add_argument("--so-texto", action="store_true")

    s = sub.add_parser("diagnosticar")
    s.add_argument("arquivos", nargs="+")

    s = sub.add_parser("comparar")
    s.add_argument("--rotulado", default="dados/rotulado")
    s.add_argument("--modelo", default=MODELO_PADRAO)

    s = sub.add_parser("demo")
    s.add_argument("--destino", default="exemplos")

    a = ap.parse_args(argv)
    dump = lambda o: print(json.dumps(o, ensure_ascii=False, indent=2, default=str))  # noqa: E731

    if a.cmd == "extrair-fontes":
        from .documento import extrair_fontes_pdf
        for p in a.pdfs:
            for f in extrair_fontes_pdf(p, a.destino):
                print("fonte salva:", f)
    elif a.cmd == "treinar":
        from .train import treinar
        treinar(a.fontes, a.rotulado, a.saida, a.passos, a.lote, a.lr, a.continuar, a.prop_real, a.frases, a.workers, avaliar_cada=a.avaliar_cada)
    elif a.cmd == "rotular":
        from .rotular import gerar
        n = gerar(a.arquivos, a.destino, a.modelo, a.tesseract)
        print(f"{n} recortes em {a.destino}/. Corrija os .gt.txt e mova para dados/rotulado/.")
    elif a.cmd == "ler":
        from .ler import ler_documento
        for f in a.arquivos:
            r = ler_documento(f, a.modelo, a.forcar_ocr)
            print(r["texto"]) if a.so_texto else dump(r)
    elif a.cmd == "diagnosticar":
        from .diagnostico import diagnosticar_arquivo
        for f in a.arquivos:
            dump(diagnosticar_arquivo(f))
    elif a.cmd == "comparar":
        from .diagnostico import comparar_motores
        dump(comparar_motores(a.rotulado, a.modelo))
    elif a.cmd == "demo":
        from .demo import gerar_demo
        gerar_demo(a.destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
