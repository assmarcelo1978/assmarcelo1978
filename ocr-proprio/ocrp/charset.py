"""Alfabeto que o modelo sabe escrever.

Mantemos um conjunto fechado e pequeno: tudo que aparece em contas,
extratos e boletos brasileiros. Quanto menor o alfabeto, menos o modelo
confunde (ex.: não existe "€" para ser confundido com "C").
"""

DIGITOS = "0123456789"
MAIUSCULAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MINUSCULAS = "abcdefghijklmnopqrstuvwxyz"
ACENTOS = "ÁÀÂÃÉÊÍÓÔÕÚÜÇáàâãéêíóôõúüç"
SIMBOLOS = " .,:;/-()$%*+=#@&ºª°_'\"!?<>[]|"

ALFABETO = DIGITOS + MAIUSCULAS + MINUSCULAS + ACENTOS + SIMBOLOS

# índice 0 é reservado para o "blank" do CTC
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALFABETO)}
IDX2CHAR = {i + 1: c for i, c in enumerate(ALFABETO)}
NUM_CLASSES = len(ALFABETO) + 1


def normalizar(texto: str) -> str:
    """Troca caracteres equivalentes e descarta o que o modelo não conhece."""
    trocas = {"–": "-", "—": "-", " ": " ", "’": "'", "“": '"', "”": '"'}
    texto = "".join(trocas.get(c, c) for c in texto)
    return "".join(c for c in texto if c in CHAR2IDX)


def codificar(texto: str) -> list[int]:
    return [CHAR2IDX[c] for c in normalizar(texto)]
