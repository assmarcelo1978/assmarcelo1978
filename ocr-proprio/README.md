# OCR próprio: contas, extratos e boletos

Um OCR que **você treina com as fontes dos seus documentos**: contas de luz,
água, extratos bancários e boletos. Ele roda no seu computador, sem gastar
token de IA, e marca como **duvidoso** tudo que não consegue confirmar, em vez
de entregar um número errado sem avisar.

---

## Por que os OCRs que você instalou erravam tanto (e "cada hora liam diferente")

Quase nunca o problema é "o OCR ser ruim". As causas reais são estas, da mais
comum para a menos comum:

| # | Causa | O que acontece | Como este projeto resolve |
|---|-------|----------------|---------------------------|
| 1 | **Fazer OCR em PDF que já tem texto** | Extrato do internet banking e fatura digital já trazem o texto exato dentro do PDF. Transformar em imagem e ler de novo só cria erro. | `ler` testa primeiro se o PDF tem texto. Se tiver, copia o texto, que fica 100% exato. |
| 2 | **Resolução baixa / foto do WhatsApp** | O WhatsApp comprime a imagem. A vírgula vira ponto, o 5 vira S, o 8 vira B. **R$ 1.234,56 vira 1234.56 ou 1.234.56** | Amplia a imagem até a letra ter cerca de 20 px de altura antes de ler. O `diagnosticar` avisa quando a foto está ruim demais. |
| 3 | **Foto torta, sombra, fundo colorido** | Conta de luz tem fundo colorido e marca d'água. Foto de celular tem sombra. Na binarização, parte das letras some. | Remove o fundo, binariza de forma adaptativa (por região) e corrige a inclinação. |
| 4 | **Tabela lida como texto corrido** | Em "DATA  HISTÓRICO  VALOR" o OCR junta as colunas e lê as linhas da tabela como `l`, `|` ou `_`. | Remove as linhas da tabela e lê cada coluna separadamente, guardando a posição. |
| 5 | **Modelo genérico nunca viu aquela fonte** | Tesseract e EasyOCR foram treinados com livros e sites. Nas fontes estreitas de conta, `0`/`O` e `1`/`l`/`I` ficam quase iguais. | **Rede neural própria (CRNN) treinada com as fontes dos seus documentos**, com alfabeto fechado (só os caracteres que aparecem em conta). |
| 6 | **"Cada hora lê diferente"** | Cada motor e cada configuração (`--psm`, idioma, DPI) muda o resultado. Uma IA (GPT/Claude) também varia de uma chamada para outra. | O modelo é fixo e determinístico: a mesma imagem sempre dá a mesma leitura. |
| 7 | **Ninguém conferia o resultado** | Um dígito errado passava sem ninguém ver. | **Validação matemática**: dígito verificador da linha digitável (mod 10 / mod 11), CPF/CNPJ, valor do boleto comparado com o total lido, saldo anterior + lançamento comparado com o saldo atual. Se não fecha, o resultado vem com `"confiavel": false`. |

> A IA acertava porque ela "entende" o contexto e corrige sozinha. Aqui essa
> correção é feita de forma barata e verificável: pré-processamento, um modelo
> especializado e regras de validação dos próprios documentos brasileiros.

### Teste medido (conta de luz fotografada torta, com sombra, borrada e comprimida em JPEG)

| Motor | Erro por caractere | Resultado |
|---|---|---|
| Tesseract "cru" (como costuma ser usado) | **12,1%** | pulou linhas inteiras e cortou o código de barras |
| Tesseract + o pré-processamento deste projeto | **0,36%** | todos os campos certos |
| OCR próprio, só 50 min de treino em CPU fraca, com fontes genéricas e **sem nenhum recorte real** | **0,90%** | todos os campos certos; errou só pontuação (`:` virou `.`) |
| Os dois juntos (padrão do `ler`) | — | os 5 trechos em que discordaram ficaram marcados como duvidosos |

Conclusão: **a maior parte do erro vinha da imagem chegar ao OCR sem
tratamento**. O modelo próprio acerta mais quando é treinado com as fontes
reais e uns 200 recortes reais corrigidos. Os trechos em que os dois motores
discordam são exatamente os que vale a pena corrigir e devolver ao treino.

Para ver o diagnóstico de **um documento seu**:

```bash
python -m ocrp diagnosticar minha_conta.jpg extrato.pdf
```

Ele mostra a altura das letras em pixels, a nitidez, o contraste, a inclinação
e a iluminação, e explica em português o que vai causar erro.

Para ver **quais trocas de letra cada motor faz** nos seus documentos (depois de
ter alguns recortes corrigidos):

```bash
python -m ocrp comparar
```

O resultado mostra a taxa de erro de cada um (Tesseract cru, Tesseract com
pré-processamento e o OCR próprio) e as trocas mais comuns. Exemplo:
`certo "," leu "."`: vírgula lida como ponto, causada por resolução baixa.

---

## Instalação

```bash
cd ocr-proprio
pip install -r requirements.txt
# opcional, só para comparar com o Tesseract:
#   Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-por
#   Windows: instalador do UB-Mannheim
```

Precisa de Python 3.10 ou mais novo. Não precisa de placa de vídeo. Com GPU
NVIDIA o treino fica cerca de 20 vezes mais rápido.

## Passo a passo para treinar com os SEUS documentos

### 1. Junte as fontes dos documentos

O melhor jeito é tirar as fontes **de dentro dos PDFs digitais** das suas
contas e extratos. São exatamente as mesmas fontes usadas na impressão:

```bash
python -m ocrp extrair-fontes contas/*.pdf extratos/*.pdf
```

Elas vão para `fontes/`. Se você só tiver foto ou papel, descubra a fonte (o
site *WhatFontIs* ajuda) e coloque o `.ttf` em `fontes/`. Arial, Helvetica,
Courier e Verdana cobrem a maioria das contas brasileiras.

### 2. (Opcional) Frases reais

Crie um `frases.txt` com textos que aparecem nos seus documentos, um por linha:
nome das tarifas, históricos do extrato ("PIX RECEBIDO", "TAR PACOTE
SERVICOS"...). O gerador usa essas frases com mais frequência.

### 3. Treine

```bash
python -m ocrp treinar --frases frases.txt
```

O treino gera linhas sintéticas com as suas fontes, já com os defeitos de foto
e scanner (borrão, ruído, JPEG, sombra, inclinação), e salva o melhor modelo em
`modelos/ocr_proprio.pt`. Tempo aproximado de 20.000 passos:

* placa de vídeo NVIDIA: 15 a 30 min
* CPU boa (8 núcleos ou mais): algumas horas (deixe rodando durante a noite)

Você pode parar quando o `CER` (taxa de erro por caractere) ficar abaixo de 1%.

### 4. Ensine com os seus documentos reais (o passo que garante o acerto)

```bash
python -m ocrp rotular fotos_de_contas/*.jpg
```

Isso recorta cada trecho de texto para `dados/para_rotular/` como `xxx.png` mais
`xxx.gt.txt`, já preenchido com o palpite do modelo. Abra os `.gt.txt`, corrija
o que estiver errado e mova o par para `dados/rotulado/`.

**Atalho sem digitar nada:** se você tem o PDF digital de uma conta, rode
`rotular` nele. O rótulo sai do texto do próprio PDF e já vem 100% correto.

Depois faça o ajuste fino a partir do modelo que já existe:

```bash
python -m ocrp treinar --continuar modelos/ocr_proprio.pt --passos 5000 --prop-real 0.6
```

Repita sempre que aparecer um tipo de documento novo. Umas 200 a 500 linhas
corrigidas por tipo de documento costumam bastar.

### 5. Use

```bash
python -m ocrp ler conta.jpg                # JSON com texto, campos, alertas
python -m ocrp ler extrato.pdf --so-texto   # só o texto
```

Exemplo de saída:

```json
{
  "metodo": "texto nativo do PDF (sem OCR, exato)",
  "campos": {
    "total": {"texto": "R$ 287,43", "valor": 287.43},
    "vencimento": {"texto": "25/11/2026"},
    "cpf": [{"texto": "529.982.247-25", "valido": true}],
    "linha_digitavel": {"tipo": "convenio", "valido": true, "valor": 287.43, "segmento": "energia/gás"}
  },
  "alertas": [],
  "confiavel": true
}
```

* O `ler` usa o modelo próprio e, se o Tesseract estiver instalado, confere
  cada trecho com ele (segunda opinião). Trechos em que os dois discordam vão
  para `trechos_duvidosos` com as duas leituras. Para desligar, use
  `--sem-tesseract`.
* `confiavel: true`: todas as validações fecharam.
* `confiavel: false`: veja `alertas` e `trechos_duvidosos`. São os únicos
  pontos que precisam de olho humano (ou, se preferir, só eles vão para a IA,
  gastando pouquíssimo token).

### Usar dentro de outro programa (bot de WhatsApp, painel...)

```python
from ocrp.ler import ler_documento
r = ler_documento("conta.jpg")
if r["confiavel"]:
    salvar(r["campos"]["total"]["valor"], r["campos"]["vencimento"]["texto"])
else:
    pedir_confirmacao(r["alertas"])
```

Para o bot em Node.js (`botfinanceirozap`), chame
`python -m ocrp ler arquivo --modelo ...` com `child_process.execFile` e leia o
JSON.

---

## Teste rápido sem documentos seus

```bash
python -m ocrp demo                                  # cria exemplos/ (PDF digital + "foto" ruim)
python -m ocrp ler exemplos/conta_luz_digital.pdf    # lê o texto nativo e valida o código de barras
python -m ocrp diagnosticar exemplos/conta_luz_foto.jpg
```

## Estrutura

```
ocrp/
  documento.py    abre PDF/imagem, texto nativo do PDF, extrai fontes embutidas
  preprocess.py   escala, remoção de fundo/sombra, binarização, correção de inclinação
  segment.py      acha linhas/colunas de texto, remove linhas de tabela
  model.py        rede CRNN + CTC (o OCR em si) e o leitor
  synth.py        gera linhas de treino com as suas fontes e defeitos de foto
  train.py        treino (sintético + recortes reais corrigidos)
  campos.py       extrai e VALIDA total, vencimento, CPF/CNPJ, linha digitável, extrato
  diagnostico.py  explica por que erra; compara motores contra o gabarito
  rotular.py      gera recortes dos seus documentos para corrigir
  ler.py          pipeline completo
```
