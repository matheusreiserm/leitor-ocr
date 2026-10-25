# Ferramentas de reconstrução de livros escaneados

Transformam o PDF escaneado de um livro em duas versões limpas: `<título> - limpo.pdf` (texto preto em
fundo branco) e `<título> - papel.pdf` (o mesmo, com textura de papel). Cada livro tem a sua pasta
`<título> - limpo/` ao lado do PDF original, em `Downloads\Livr`.

Livros já tratados: **La Serpiente**, **Invención del Tren Fantasma**, **La Vida Nueva**, **Biografía**, **Berkeley em Bellagio**.

`preparar.py` reconhece dois tipos de PDF:
- **escaneamento em páginas duplas, preto e branco** (La Vida Nueva): limpeza agressiva de sujeira, lascas e riscos;
- **fotos de uma página por folha, em cinza** (Biografía): fundo branco por divisão do fundo estimado,
  correção de curvatura e de perspectiva, e limpeza conservadora (fica toda a tinta dentro da mancha de texto).
  Nesses livros, **não rode a etapa 3** (manchas): as fotos já são limpas e ela pode apagar pontuação.

- **escaneamento já legível, em cor** (Berkeley em Bellagio, em português): **não limpe**. Use
  `fatiar.py "<livro.pdf>" "<pasta>" "<título>"`, que só corta cada folha nas duas páginas, sem endireitar, clarear
  nem realinhar, põe o texto do OCR invisível por baixo (busca e seleção) e as capas do `config.json` antes e
  depois. A limpeza (`preparar.py` em modo cinza, etapa 3, triagem) apagou letras claras perto da lombada e só
  piorou o livro. Reconstruir só onde a impressão está apagada e o OCR não alcança.

### Configuração por livro (`<pasta>/reconstrucao/config.json`)

```json
{"geometria": 2, "idioma": "por", "ignorar": [[1, "esq"], [54, "*"]], "papel": false,
 "capas": {"antes": [{"folha": 54, "caixa": [1730, 0, 3316, 2460], "nome": "capa"}],
           "depois": [{"folha": 54, "caixa": [0, 0, 1600, 2460], "nome": "contracapa"}]}}
```

- `idioma`: `spa` (padrão) ou `por`. Em português, valem o léxico `pt.json.gz`, o Tesseract `por`, os tis em
  ã/õ e a grafia anterior a 2009 ("idéia", "vôo", "freqüente", "pára"), que o léxico atual não traz (`idioma.py`).
- `ignorar`: folhas do PDF (ou só uma metade, `esq`/`dir`) fora do miolo, como capas e orelhas.
- `capas`: recortes do próprio escaneamento (`capas.py`), postos antes e depois do miolo no PDF, em cor e sem
  textura. Os nomes com "capa" ou "orelha" ficam sem textura; os outros ("rosto") a recebem.
- `papel`: `false` gera só o PDF limpo (para livro cujo papel tem cor própria, a textura não acrescenta nada).

## Instalação

Python 3 com `opencv-python numpy img2pdf pymupdf pyspellchecker` e o
[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) em `C:\Program Files\Tesseract-OCR`.
Dois arquivos grandes não vão no repositório; baixe-os para dentro desta pasta:

- `tessdata\spa.traineddata`: <https://github.com/tesseract-ocr/tessdata_best/raw/main/spa.traineddata>.
  Copie também `osd.traineddata` e a pasta `configs` de `C:\Program Files\Tesseract-OCR\tessdata`.
- `lexico\es_full.txt`: <https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_full.txt>.
  Depois gere `lexico\es.json.gz` (palavras com frequência ≥ 3):
  `py -c "import json,gzip; d={w:int(n) for w,n in (l.rsplit(' ',1) for l in open('lexico/es_full.txt',encoding='utf-8')) if int(n)>=3 and w.isalpha()}; json.dump(d,gzip.open('lexico/es.json.gz','wt',encoding='utf-8'),ensure_ascii=False)"`

## As etapas

Rode tudo de dentro desta pasta, com `py -I <script> ...` (e `$env:PYTHONIOENCODING='utf-8'` no PowerShell).

| Etapa | Comando | O que faz |
|---|---|---|
| 1. Limpar páginas | `preparar.py "<livro.pdf>" "<pasta>"` | Separa as páginas duplas, endireita, tira bordas, sombra da lombada, riscos verticais do escaneamento, sujeira nas margens e entre linhas, e amplia 1,5×. Anotações à mão: `<pasta>\apagar.json`. |
| 2. OCR | `ocr.py "<pasta>"` | Tesseract em espanhol (`tessdata\spa`, versão *best*) → `ocr.json`. |
| 3. Manchas | `triagem.py "<pasta>" --limpar` e de novo `ocr.py` (duas vezes) | Apaga a sujeira grudada nas palavras quando é certo que é sujeira (marca sobre letra que não leva acento, traço sob a linha, marca solta no espaço entre duas palavras que o OCR juntou). Guarda as páginas anteriores em `paginas_antes_manchas\` e registra tudo em `reconstrucao\manchas_removidas.json`. |
| 4. Limpeza julgada | `limpar_manual.py "<pasta>"` e `ocr.py` | Aplica `reconstrucao\limpeza_manual.json` (casos olhados um a um). |
| 5. Triagem | `triagem.py "<pasta>"` | Lista as palavras a rever → `reconstrucao\triagem.json`. |
| 6. Propostas | `propor.py "<pasta>"` | Gera `reconstrucao\propostas.json` (regras automáticas + `reconstrucao\julgamento.json`) e `pendentes.json` (o que ainda falta julgar). |
| 7. Letras | `py -I -c "import sys; sys.path.insert(0,'.'); from glifos import construir; construir(r'<pasta>')"` | Biblioteca de letras recortadas do próprio livro (`glifos.pkl`). |
| 8. Revisão | `.\revisar.ps1 serpiente\|tren\|vida` | Você decide caso a caso, vendo ANTES/DEPOIS. Enter aceita, `n` mantém, digitar corrige, `A` aceita de uma vez os de certeza alta, `q` sai (as decisões ficam salvas). |
| 9. Aplicar | `.\aplicar.ps1 serpiente\|tren\|vida` | Redesenha as palavras aprovadas e refaz os dois PDFs. |

## Arquivos de decisão (por livro, em `reconstrucao\`)

- `julgamento.json` e `limpeza_manual.json` citam os ids de `triagem_julgada.json` (a triagem sobre a qual
  se julgou). Os ids mudam a cada rodada; os scripts ligam as decisões aos casos atuais **pela posição na
  página**, então dá para refazer OCR e triagem sem perder o julgamento.
- `config.json` com `{"geometria": 2}` liga a geometria nova da reconstrução (reconhece pontuação pela forma,
  não invade palavras vizinhas). Sem esse arquivo (La Serpiente, Tren), vale a geometria original.
- `decisoes.json` é escrito pela revisão (etapa 8).
