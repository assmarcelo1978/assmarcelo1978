"""Gera imagens de treino sintéticas com AS FONTES DOS SEUS DOCUMENTOS.

Ideia central: o OCR genérico erra porque nunca viu a fonte da sua conta de
luz/extrato. Aqui renderizamos milhares de linhas parecidas com as dos
documentos reais (datas, valores R$, CPF/CNPJ, códigos de barras, termos de
conta de energia e extrato) usando os arquivos .ttf/.otf que você colocar em
`fontes/`, e simulamos os defeitos de scanner/foto (borrão, ruído, JPEG,
inclinação, tinta fraca). O modelo aprende exatamente aquilo que vai ler.
"""
from __future__ import annotations

import glob
import io
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .charset import normalizar

PALAVRAS = """
CONTA DE ENERGIA ELÉTRICA NOTA FISCAL FATURA VENCIMENTO TOTAL A PAGAR
CONSUMO kWh LEITURA ANTERIOR ATUAL MEDIDOR CLASSE RESIDENCIAL MONOFÁSICO
BIFÁSICO TRIFÁSICO BANDEIRA TARIFÁRIA VERDE AMARELA VERMELHA ICMS PIS COFINS
CONTRIB ILUM PÚBLICA MUNICIPAL TARIFA TUSD TE DEMANDA REATIVO INSTALAÇÃO
UNIDADE CONSUMIDORA CLIENTE CÓDIGO REFERÊNCIA MÊS EMISSÃO APRESENTAÇÃO
EXTRATO CONTA CORRENTE AGÊNCIA SALDO ANTERIOR DISPONÍVEL BLOQUEADO
LANÇAMENTOS HISTÓRICO DOCUMENTO VALOR DATA CRÉDITO DÉBITO PIX ENVIADO
RECEBIDO TED DOC TARIFA PACOTE SERVIÇOS PAGAMENTO BOLETO TRANSFERÊNCIA
SAQUE DEPÓSITO COMPRA CARTÃO DÉBITO CRÉDITO RENDIMENTO APLICAÇÃO RESGATE
POUPANÇA JUROS IOF IMPOSTO ESTORNO ENCARGOS LIMITE CHEQUE ESPECIAL
ÁGUA ESGOTO GÁS TELEFONE INTERNET CONDOMÍNIO ALUGUEL IPTU IPVA SEGURO
Banco Itaú Bradesco Santander Caixa Nubank Inter Sicoob Sicredi Brasil
Rua Avenida Travessa Bairro Centro CEP Cidade Estado São Paulo Rio de Janeiro
Minas Gerais Paraná Bahia Pernambuco Ceará Goiás Ltda S.A. ME EPP
pagamento recebido enviado transferência saldo do dia total geral
""".split()

NOMES = "MARIA JOSE ANA JOAO ANTONIO FRANCISCO CARLOS PAULO PEDRO LUCAS LUIZ MARCOS MARCELO GABRIEL RAFAEL DANIEL".split()
SOBRENOMES = "SILVA SANTOS OLIVEIRA SOUZA RODRIGUES FERREIRA ALVES PEREIRA LIMA GOMES COSTA RIBEIRO MARTINS CARVALHO ASSIS".split()


def _valor():
    v = random.choice([random.uniform(0, 99), random.uniform(0, 9999), random.uniform(0, 999999)])
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return random.choice([s, "R$ " + s, "R$" + s, "-" + s, s + " D", s + " C", s + "-"])


def _data():
    d, m, a = random.randint(1, 28), random.randint(1, 12), random.randint(2018, 2030)
    meses = "JAN FEV MAR ABR MAI JUN JUL AGO SET OUT NOV DEZ".split()
    return random.choice([f"{d:02d}/{m:02d}/{a}", f"{d:02d}/{m:02d}", f"{d:02d}/{m:02d}/{a % 100:02d}", f"{meses[m-1]}/{a}", f"{d:02d} {meses[m-1]}"])


def _cpf():
    n = [random.randint(0, 9) for _ in range(11)]
    return random.choice(["{}{}{}.{}{}{}.{}{}{}-{}{}".format(*n), "***.{}{}{}.{}{}{}-**".format(*n[:6]), "".join(map(str, n))])


def _cnpj():
    n = [random.randint(0, 9) for _ in range(14)]
    return "{}{}.{}{}{}.{}{}{}/{}{}{}{}-{}{}".format(*n)


def _linha_digitavel():
    g = lambda k: "".join(str(random.randint(0, 9)) for _ in range(k))  # noqa: E731
    return random.choice([
        f"{g(5)}.{g(5)} {g(5)}.{g(6)} {g(5)}.{g(6)} {g(1)} {g(14)}",
        f"{g(11)}-{g(1)} {g(11)}-{g(1)} {g(11)}-{g(1)} {g(11)}-{g(1)}",
        g(random.randint(6, 20)),
    ])


def _numero():
    return random.choice([str(random.randint(0, 99999)), f"{random.randint(0, 9999)} kWh", f"{random.uniform(0, 2):.6f}".replace(".", ","), f"{random.randint(0,100)}%"])


def _palavras(k):
    ws = random.choices(PALAVRAS, k=k)
    estilo = random.random()
    if estilo < 0.4:
        ws = [w.upper() for w in ws]
    elif estilo < 0.6:
        ws = [w.capitalize() for w in ws]
    return " ".join(ws)


def texto_aleatorio(extra: list[str] | None = None) -> str:
    """Uma linha 'parecida' com contas e extratos."""
    if extra and random.random() < 0.35:
        return random.choice(extra)
    tipo = random.random()
    if tipo < 0.25:
        partes = [_data(), _palavras(random.randint(1, 4)), _valor()]
    elif tipo < 0.40:
        partes = [_valor()]
    elif tipo < 0.50:
        partes = [_data()]
    elif tipo < 0.58:
        partes = [random.choice(["CPF:", "CPF", "CNPJ:", "CNPJ"]), random.choice([_cpf(), _cnpj()])]
    elif tipo < 0.65:
        partes = [_linha_digitavel()]
    elif tipo < 0.75:
        partes = [_palavras(random.randint(1, 3)), _numero()]
    elif tipo < 0.82:
        partes = [random.choice(NOMES), random.choice(SOBRENOMES), random.choice(SOBRENOMES)]
    elif tipo < 0.90:
        partes = [_palavras(random.randint(1, 6))]
    else:
        partes = ["".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(random.randint(3, 12)))]
    return normalizar(" ".join(partes)).strip() or "0"


def listar_fontes(pasta: str) -> list[str]:
    fs = []
    for ext in ("ttf", "otf", "TTF", "OTF"):
        fs += glob.glob(os.path.join(pasta, "**", f"*.{ext}"), recursive=True)
    return sorted(set(fs))


def _augmentar(img: np.ndarray) -> np.ndarray:
    """Simula scanner, foto de celular, impressora matricial/jato de tinta fraco."""
    h, w = img.shape
    if random.random() < 0.3:  # tinta mais grossa/fina
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if random.random() < 0.5 else cv2.dilate(img, k)
    if random.random() < 0.4:  # inclinação leve
        ang = random.uniform(-1.5, 1.5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
        img = cv2.warpAffine(img, M, (w, h), borderValue=int(np.median(img)))
    if random.random() < 0.5:  # baixa resolução
        f = random.uniform(0.35, 0.8)
        pequeno = cv2.resize(img, (max(4, int(w * f)), max(4, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(pequeno, (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 1.2))
    if random.random() < 0.5:
        ruido = np.random.normal(0, random.uniform(3, 18), img.shape)
        img = np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    if random.random() < 0.4:  # compressão do WhatsApp
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=random.randint(15, 70))
        img = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("L"))
    if random.random() < 0.3:  # iluminação desigual (foto)
        grad = np.linspace(random.uniform(0.6, 1), random.uniform(0.6, 1), w)[None, :]
        img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    return img


def renderizar(texto: str, fonte: str, augment: bool = True) -> np.ndarray:
    tam = random.randint(22, 40)
    font = ImageFont.truetype(fonte, tam)
    x0, y0, x1, y1 = font.getbbox(texto)
    pad_x, pad_y = random.randint(2, 12), random.randint(2, 8)
    W, H = x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y
    fundo = random.randint(200, 255)
    tinta = random.randint(0, 90)
    im = Image.new("L", (max(W, 4), max(H, 4)), fundo)
    d = ImageDraw.Draw(im)
    espac = random.choice([0, 0, 0, 1, 2]) if random.random() < 0.3 else 0
    if espac:
        x = pad_x - x0
        for c in texto:
            d.text((x, pad_y - y0), c, font=font, fill=tinta)
            x += font.getlength(c) + espac
        im = im.crop((0, 0, min(int(x) + pad_x, im.width), im.height)) if x < im.width else im
    else:
        d.text((pad_x - x0, pad_y - y0), texto, font=font, fill=tinta)
    arr = np.array(im)
    return _augmentar(arr) if augment else arr


def amostra(fontes: list[str], extra: list[str] | None = None) -> tuple[np.ndarray, str]:
    while True:
        t = texto_aleatorio(extra)
        f = random.choice(fontes)
        try:
            return renderizar(t, f), t
        except Exception:
            continue
