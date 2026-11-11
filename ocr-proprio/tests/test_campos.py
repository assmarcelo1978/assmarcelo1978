"""Testes das validações (rodar: python -m pytest tests  ou  python tests/test_campos.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocrp import campos  # noqa: E402
from ocrp.demo import linha_convenio_energia  # noqa: E402
from ocrp.metrics import cer, confusoes  # noqa: E402


def _boleto(banco="341", fator=1500, valor_centavos=12345, livre="1234567890123456789012345"):
    sem_dv = banco + "9" + f"{fator:04d}{valor_centavos:010d}" + livre
    dv = campos.mod11_boleto(sem_dv)
    barras = sem_dv[:4] + str(dv) + sem_dv[4:]
    c1 = barras[0:4] + barras[19:24]
    c2 = barras[24:34]
    c3 = barras[34:44]
    return f"{c1}{campos.mod10(c1)}{c2}{campos.mod10(c2)}{c3}{campos.mod10(c3)}{barras[4]}{barras[5:19]}"


def test_boleto_valido_e_valor():
    ld = _boleto()
    r = campos.validar_linha_digitavel(ld)
    assert r["valido"] and r["tipo"] == "boleto" and abs(r["valor"] - 123.45) < 1e-9


def test_boleto_um_digito_errado_e_detectado():
    ld = _boleto()
    for i in range(47):
        errado = ld[:i] + str((int(ld[i]) + 1) % 10) + ld[i + 1:]
        assert not campos.validar_linha_digitavel(errado)["valido"], i


def test_convenio_energia():
    ld = linha_convenio_energia(287.43)
    r = campos.validar_linha_digitavel(ld)
    assert r["valido"] and r["segmento"] == "energia/gás" and abs(r["valor"] - 287.43) < 1e-9
    errado = ld[:5] + str((int(ld[5]) + 1) % 10) + ld[6:]
    assert not campos.validar_linha_digitavel(errado)["valido"]


def test_letras_viram_digitos_em_campo_numerico():
    ld = _boleto().replace("0", "O").replace("1", "l")
    assert campos.validar_linha_digitavel(ld)["valido"]


def test_cpf_cnpj():
    assert campos.cpf_valido("529.982.247-25")
    assert not campos.cpf_valido("529.982.247-26")
    assert campos.cnpj_valido("11.222.333/0001-81")
    assert not campos.cnpj_valido("11.222.333/0001-82")


def test_total_diferente_do_codigo_de_barras_gera_alerta():
    txt = f"TOTAL A PAGAR R$ 287,48\n{linha_convenio_energia(287.43)}"
    r = campos.extrair(txt)
    assert not r["confiavel"] and r["campos"]["total"]["corrigido_por_codigo_barras"] == 287.43


def test_extrato_saldo():
    txt = "01/10 SALDO ANTERIOR 1.000,00 1.000,00\n02/10 PIX RECEBIDO 250,00 1.250,00\n03/10 TAR PACOTE 19,90 D 1.230,10\n04/10 COMPRA 30,00 D 1.200,10"
    lanc = campos.lancamentos_extrato(txt)
    assert len(lanc) == 4 and lanc[2]["valor"] == -19.90
    assert campos.conferir_saldos(lanc) == []
    lanc[3]["valor"] = -3.00  # leitura errada de 30,00
    assert campos.conferir_saldos(lanc)


def test_metricas():
    assert cer("R$ 1.234.56", "R$ 1.234,56") == 1 / 11
    assert confusoes([("1O", "10")])[("0", "O")] == 1


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok", nome)
