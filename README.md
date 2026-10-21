# leitor-ocr

Limpeza de páginas escaneadas guiada pelo próprio livro, antes do OCR, e comparação de motores de OCR.
Complementa [reconstrucao-livros](https://github.com/matheusreiserm/reconstrucao-livros), que precisa estar
clonado **ao lado** desta pasta, com o nome `ferramentas_reconstrucao` (os scripts usam o `ocr.py`, a `triagem.py`
e o léxico de lá).

## A ideia

Neste tipo de livro, a sujeira tem o tamanho de um acento ou de um ponto, então filtros que só olham o tamanho
(mediana, Otsu) não separam as duas coisas. A posição separa:

1. **Linhas, pelo OCR.** A linha de base e a altura de x de cada linha vêm das palavras do Tesseract, não da
   projeção da página.
2. **Léxico.** Cada palavra é corrigida pelo léxico ("estabá" → "estaba", "eñañitos" → "enanitos"). Palavras
   ambíguas ("esta/está", "el/él") não são tocadas.
3. **Moldes de letra** (agrupamento de símbolos). As letras de palavras confirmadas são alinhadas pela linha de
   base e somadas em um molde por caractere (39 no La Vida Nueva). O molde diz onde fica o acento de cada letra:
   o do "í", por exemplo, fica um pouco à direita da haste.
4. **Julgamento de cada peça pequena de tinta:**
   - entre linhas: sujeira;
   - acima da letra: fica só a marca que a palavra pede, no ponto que o molde indica;
   - abaixo de letra sem perna: sujeira.

   Como as linhas são próximas, primeiro cada linha protege as marcas que pede, e só depois se apaga o resto.
   A limpeza nunca acrescenta tinta.

`moldes.py` também tenta apagar sujeira **grudada** nas letras (o que o molde não explica). Por enquanto, para
não cortar pontas de letra, essa camada ficou tão cautelosa que quase não altera o resultado.

## Resultado (La Vida Nueva, 10 páginas, gabarito do Surya, 2.517 palavras)

| Variante | Tesseract: fora do léxico | Tesseract: erro vs Surya | indra: erro vs Surya |
|---|---|---|---|
| Página bruta (saída do `preparar.py`) | 1,91% | 2,11% | 3,02% |
| Fluxo atual (manchas + limpeza manual) | 1,83% | 1,83% | 1,91% |
| Mediana 3×3 | 1,79% | 1,83% | 2,54% |
| Otsu | 2,03% | 2,30% | 2,86% |
| Linhas + léxico sobre a bruta | 1,59% | 1,83% | 2,19% |
| **Fluxo atual + linhas + léxico** | **1,43%** | **1,51%** | **1,55%** |
| … + moldes contra sujeira grudada | 1,63% | 1,55% | 1,51% |

## Uso

Rode tudo de dentro de uma **pasta de trabalho** (não desta), com `py -I -X utf8`. Os caches (`_moldes.pkl`,
`_indra/`) ficam nela.

```
py -I -X utf8 <repo>\moldes.py "<livro>\paginas_antes_manchas" <entrada> <saida_moldes>   # gera _moldes.pkl
py -I -X utf8 <repo>\limpar_por_moldes.py <entrada> <saida>                               # limpeza por linhas + léxico
py -I -X utf8 <repo>\medir.py <variante1> <variante2> ...                                 # precisa de gabarito_surya/
py -I -X utf8 <repo>\mudancas.py <variante_a> <variante_b>                                # o que mudou, com recortes em mudou/
```

`limpar_por_moldes.py` grava, ao lado de cada página limpa, `apagado/<pagina>.png` com o que foi apagado em
vermelho. `comparar.py` mede a parte das palavras fora do léxico em textos de OCR.

## Motores de OCR

| Motor | Uso | Tempo por página (CPU) |
|---|---|---|
| Tesseract (`spa`/`por` *best*) | volume, e o único com posição de cada letra | < 1 s |
| indra-ocr | volume, segunda opinião | ~0,5 s |
| Surya 2 | casos difíceis e gabarito | ~3 min |
| PaddleOCR | reserva (sem MKLDNN, lento) | ~5 min |
| EasyOCR, docTR | descartados (trocas t/r; docTR perde os acentos) | 5–50 s |

Cada motor pesado tem o seu ambiente virtual, para não conflitar com o opencv/numpy das ferramentas:

```
C:\Users\<você>\ocr_motores\surya     Python 3.13 + torch CPU + surya-ocr
C:\Users\<você>\ocr_motores\paddle    Python 3.13 + paddlepaddle (índice CPU do Paddle) + paddleocr
C:\Users\<você>\ocr_motores\doctr     Python 3.14 + torch CPU + python-doctr[torch]
C:\Users\<você>\ocr_motores\easyocr   Python 3.14 + torch CPU + easyocr
```

- **Surya 2** usa o `llama-server` do llama.cpp (`winget install ggml.llamacpp`). Antes de rodar, defina
  `LLAMA_CPP_BINARY` com o caminho dele. Em CPU, use `SURYA_INFERENCE_PARALLEL=2` e
  `SURYA_INFERENCE_TIMEOUT_SECONDS=3600`; com o padrão (8 em paralelo), as páginas estouram o tempo.
- **PaddleOCR** no Windows precisa de `enable_mkldnn=False`, porque o paddlepaddle 3.4 falha no oneDNN.

`aquecer/` tem um script por motor que baixa os modelos e lê uma página de teste.
