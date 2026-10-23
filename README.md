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
   ambíguas ("esta/está", "el/él") não são tocadas, e um -ó final também não ("aleteó", "moduló"). O léxico
   vem de legendas e não traz pretéritos raros; sem essa regra, o acento verdadeiro era tratado como sujeira.
   Ficam de fora dessa proteção gerúndio, particípio e pronome depois de infinitivo ("parecidó", "taparló"),
   que nunca levam esse acento.
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

## Versão final de um livro já revisado

Quando a revisão das reconstruções (etapa 8 do reconstrucao-livros) já está feita, a limpeza nova entra sem
refazer OCR, triagem nem revisão:

```
py -I -X utf8 <repo>\limpar_por_moldes.py "<livro>\paginas_sem_reconstrucao" base_nova     # na pasta de trabalho, com _moldes.pkl
copie base_nova\*.png para "<livro>\paginas_sem_reconstrucao" (guarde antes a versão anterior)
py -I -X utf8 <repo>\aplicar_sobre_limpa.py "<livro>" "<cópia da base anterior>" "<livro>\paginas_sem_reconstrucao"
py -I <ferramentas>\pdfs_livro.py <ferramentas> "<livro>" "<título>"
```

`aplicar_sobre_limpa.py` mede a geometria de cada palavra aprovada na página-base **anterior**, a mesma da
revisão, e desenha sobre a página limpa. Por isso as palavras redesenhadas saem idênticas às aprovadas. Medir na
página limpa encolhia a área apagada em volta de algumas palavras e deixava restos ao lado ("era 'haber").
O script também grava as páginas sem reconstrução, que o `aplicar.py` deixa como estavam. A conferência é feita
por pixel: a versão nova só pode diferir da anterior onde a limpeza apagou tinta.

## Avaliação

```
py -I -X utf8 <repo>\exportar_textos.py ocr <pasta_de_paginas> textos\<variante>        # tesseract\ e indra\, um .txt por página
py -I -X utf8 <repo>\exportar_textos.py surya <results.json> [...] --saida referencia   # gabarito do Surya em .txt
py -I -X utf8 <repo>\avaliar_gabarito.py --referencias referencia --hipoteses textos\<variante>\tesseract --saida aval\<nome>
py -I -X utf8 <repo>\texto_consenso.py referencia textos\<v>\tesseract textos\<v>\indra "<título> - texto.md" "<título>"
```

`avaliar_gabarito.py` (da revisão técnica 01) dá WER, CER e as substituições que diferem só por acento, além de
`divergencias.csv` palavra a palavra. Ele aceita qualquer referência em `.txt`. Com o Surya, mede concordância com
outro OCR; com páginas transcritas à mão, mede fidelidade ao impresso. Os testes dele rodam sem `-I`
(`py -E -s -X utf8 testes_avaliacao.py`), porque importam o módulo da mesma pasta.

`texto_consenso.py` gera o texto do livro a partir da leitura do Surya. Uma palavra só é trocada quando ela não
está no léxico e Tesseract e indra leram, no mesmo lugar, a mesma palavra do léxico. Cada troca vai para um `.csv`.

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
  Em livros inteiros, rode em lotes de ~10 páginas: o Surya só grava o resultado no fim de cada execução, e se
  o computador dormir no meio, a conexão com o servidor fica pendurada e tudo se perde. No Windows, impeça o
  sono durante a execução (`SetThreadExecutionState`), sem mudar as opções de energia.
- **PaddleOCR** no Windows precisa de `enable_mkldnn=False`, porque o paddlepaddle 3.4 falha no oneDNN.

`aquecer/` tem um script por motor que baixa os modelos e lê uma página de teste.
