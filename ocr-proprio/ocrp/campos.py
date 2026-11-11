"""Extrai campos (vencimento, total, CPF/CNPJ, linha digitável...) e VALIDA.

OCR nunca é 100% sozinho. O que dá a garantia de "sempre acerta" é a
validação: boleto e convênio têm dígitos verificadores, CPF/CNPJ também,
e o valor do boleto está codificado dentro da linha digitável. Se a
leitura não fecha a conta, o campo é marcado como DUVIDOSO em vez de ser
aceito errado em silêncio.
"""
from __future__ import annotations

import re
from datetime import date

# ---------------------------------------------------------------- correções
# Em campos numéricos, letras parecidas quase sempre são dígitos mal lidos.
TROCA_NUMERICA = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                                "i": "1", "Z": "2", "S": "5", "s": "5", "B": "8", "G": "6", "b": "6", "g": "9", "q": "9", "T": "7"})


def so_digitos(s: str) -> str:
    return re.sub(r"\D", "", s.translate(TROCA_NUMERICA))


# ---------------------------------------------------------------- validadores
def mod10(num: str) -> int:
    soma, peso = 0, 2
    for d in reversed(num):
        p = int(d) * peso
        soma += p // 10 + p % 10
        peso = 1 if peso == 2 else 2
    return (10 - soma % 10) % 10


def mod11_boleto(num: str) -> int:
    soma, peso = 0, 2
    for d in reversed(num):
        soma += int(d) * peso
        peso = 2 if peso == 9 else peso + 1
    r = 11 - soma % 11
    return 1 if r in (0, 10, 11) else r


def mod11_convenio(num: str) -> int:
    soma, peso = 0, 2
    for d in reversed(num):
        soma += int(d) * peso
        peso = 2 if peso == 9 else peso + 1
    r = soma % 11
    return 0 if r in (0, 1) else 11 - r


def cpf_valido(s: str) -> bool:
    d = so_digitos(s)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for n in (9, 10):
        soma = sum(int(d[i]) * (n + 1 - i) for i in range(n))
        if (soma * 10 % 11) % 10 != int(d[n]):
            return False
    return True


def cnpj_valido(s: str) -> bool:
    d = so_digitos(s)
    if len(d) != 14 or d == d[0] * 14:
        return False
    for n, pesos in ((12, [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]), (13, [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])):
        r = sum(int(d[i]) * pesos[i] for i in range(n)) % 11
        if (0 if r < 2 else 11 - r) != int(d[n]):
            return False
    return True


def validar_linha_digitavel(s: str) -> dict:
    """Boleto bancário (47 dígitos) ou conta de consumo/convênio (48 dígitos, começa com 8)."""
    d = so_digitos(s)
    out = {"digitos": d, "valido": False}
    if len(d) == 47:
        campos_ok = all(mod10(d[a:b - 1]) == int(d[b - 1]) for a, b in ((0, 10), (10, 21), (21, 32)))
        barras = d[0:4] + d[32] + d[33:47] + d[4:9] + d[10:20] + d[21:31]
        dv_ok = mod11_boleto(barras[:4] + barras[5:]) == int(barras[4])
        fator, valor = int(d[33:37]), int(d[37:47]) / 100
        out.update(tipo="boleto", valido=campos_ok and dv_ok, valor=valor, campos_ok=campos_ok, dv_geral_ok=dv_ok)
        if fator >= 1000:
            from datetime import timedelta
            # o fator de vencimento reiniciou em 22/02/2025 (voltou a 1000);
            # escolhe a interpretação mais próxima de hoje
            cands = [date(1997, 10, 7) + timedelta(days=fator), date(2025, 2, 22) + timedelta(days=fator - 1000)]
            venc = min(cands, key=lambda c: abs((c - date.today()).days))
            out["vencimento"] = venc.strftime("%d/%m/%Y")
    elif len(d) == 48 and d[0] == "8":
        modulo = mod10 if d[2] in "67" else mod11_convenio
        blocos = [d[i:i + 12] for i in range(0, 48, 12)]
        ok = all(modulo(b[:11]) == int(b[11]) for b in blocos)
        out.update(tipo="convenio", valido=ok, valor=int("".join(b[:11] for b in blocos)[4:15]) / 100,
                   segmento={"1": "prefeitura", "2": "saneamento", "3": "energia/gás", "4": "telecom", "5": "órgão público", "6": "outros", "7": "multa trânsito"}.get(d[1], "?"))
    return out


# ---------------------------------------------------------------- extração
RE_VALOR = re.compile(r"(?:R\$\s*)?(-?\d{1,3}(?:\.\d{3})*,\d{2}|-?\d+,\d{2})")
RE_DATA = re.compile(r"\b(\d{2})/(\d{2})/(\d{2,4})\b")
# formatado (com pontuação) ou, sem pontuação, só logo depois do rótulo "CPF"/"CNPJ"
# — evita confundir pedaços do código de barras com CPF
RE_CPF = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|(?<=CPF)[:\s]*\d{11}\b", re.I)
RE_CNPJ = re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b|(?<=CNPJ)[:\s]*\d{14}\b", re.I)
RE_LINHA = re.compile(r"[\dOoIlSB][\dOoIlSB .\-]{40,70}[\dOoIlSB]")


def valor_float(s: str) -> float:
    return float(s.replace("R$", "").replace(" ", "").replace(".", "").replace(",", "."))


def _perto(linhas: list[str], chaves: tuple[str, ...], regex: re.Pattern, janela: int = 2):
    for i, ln in enumerate(linhas):
        up = ln.upper()
        if any(k in up for k in chaves):
            for j in range(i, min(len(linhas), i + janela + 1)):
                trecho = linhas[j][up.index(next(k for k in chaves if k in up)):] if j == i else linhas[j]
                m = regex.search(trecho)
                if m:
                    return m.group(0), j
    return None, None


def extrair(texto: str) -> dict:
    linhas = [l for l in texto.splitlines() if l.strip()]
    res: dict = {"campos": {}, "alertas": []}
    c = res["campos"]

    v, ln = _perto(linhas, ("TOTAL A PAGAR", "VALOR A PAGAR", "VALOR DO DOCUMENTO", "TOTAL DA FATURA", "VALOR TOTAL", "VALOR COBRADO"), RE_VALOR)
    if v:
        c["total"] = {"texto": v, "valor": valor_float(v), "linha": ln}
    d, ln = _perto(linhas, ("VENCIMENTO", "VENC."), RE_DATA)
    if d:
        c["vencimento"] = {"texto": d, "linha": ln}
    for rx, nome, valida in ((RE_CNPJ, "cnpj", cnpj_valido), (RE_CPF, "cpf", cpf_valido)):
        achados = []
        for m in rx.finditer(texto):
            achados.append({"texto": m.group(0).strip(": \t"), "valido": valida(m.group(0))})
        if achados:
            c[nome] = achados
    for m in RE_LINHA.finditer(texto):
        info = validar_linha_digitavel(m.group(0))
        if len(info["digitos"]) in (47, 48):
            c["linha_digitavel"] = info
            break

    # ---- verificações cruzadas
    ld = c.get("linha_digitavel")
    parece_conta = "total" in c or "vencimento" in c
    if parece_conta and not ld:
        res["alertas"].append("Parece conta/boleto mas não achei a linha digitável completa: o OCR pode ter cortado ou pulado texto.")
    if ld and not ld["valido"]:
        res["alertas"].append("Linha digitável com dígito verificador errado: algum número foi mal lido.")
    if ld and ld["valido"] and "total" in c and ld.get("valor") and abs(ld["valor"] - c["total"]["valor"]) > 0.005:
        res["alertas"].append(f"Total lido ({c['total']['texto']}) difere do valor codificado no código de barras (R$ {ld['valor']:.2f}). Use o do código de barras.")
        c["total"]["corrigido_por_codigo_barras"] = ld["valor"]
    for nome in ("cpf", "cnpj"):
        for item in c.get(nome, []):
            if not item["valido"]:
                res["alertas"].append(f"{nome.upper()} {item['texto']} não passa no dígito verificador.")
    res["confiavel"] = not res["alertas"]
    return res


def lancamentos_extrato(texto: str) -> list[dict]:
    """Extrai linhas do tipo  DD/MM  HISTÓRICO  VALOR  [SALDO]."""
    out = []
    for ln in texto.splitlines():
        m = re.match(r"\s*(\d{2}/\d{2}(?:/\d{2,4})?)\s+(.+?)\s+(-?(?:\d{1,3}(?:\.\d{3})*|\d+),\d{2})\s*([DC-])?(?:\s+(-?(?:\d{1,3}(?:\.\d{3})*|\d+),\d{2}))?\s*$", ln)
        if m:
            val = valor_float(m.group(3))
            if m.group(4) in ("D", "-"):
                val = -abs(val)
            out.append({"data": m.group(1), "historico": m.group(2).strip(), "valor": val, "saldo": valor_float(m.group(5)) if m.group(5) else None})
    return out


def conferir_saldos(lanc: list[dict]) -> list[str]:
    """saldo_anterior + valor == saldo_atual. Se não bate, algum número foi mal lido."""
    alertas, ult = [], None
    for l in lanc:
        if l["saldo"] is not None:
            if ult is not None and abs(ult + l["valor"] - l["saldo"]) > 0.01:
                alertas.append(f"{l['data']} {l['historico']}: saldo não fecha ({ult:.2f} + {l['valor']:.2f} != {l['saldo']:.2f})")
            ult = l["saldo"]
    return alertas
