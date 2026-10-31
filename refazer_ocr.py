"""Refaz só a camada de texto de um PDF escaneado que já tem OCR ruim (acentos trocados: "crianga", "néo").
As imagens das páginas são copiadas sem recompressão; o texto antigo é descartado e o novo, do Tesseract
(com a mesma escolha por página e as correções do ocr_pdf.py), fica invisível por baixo.

Uso: py refazer_ocr.py "<livro.pdf>" "<saida.pdf>" [--idioma por] [--ampliar 2] [--grafia-antiga] [--sem-lixo 0.2]
Escaneamento de baixa resolução (~150 dpi) é lido ampliado (--ampliar), o que o Tesseract lê melhor.
"""
import sys, tempfile, collections
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pymupdf
import idioma, ocr as O, triagem as T
from preparar import fundo_branco
from ocr_pdf import camada_texto, juntar_marcas, unicode_certo


def imagem_da_pagina(doc, pg):
    """A maior imagem da página: (bytes originais, extensão, caixa na página, largura em px)."""
    infos = sorted(pg.get_image_info(xrefs=True), key=lambda i: -(i['bbox'][2] - i['bbox'][0]) * (i['bbox'][3] - i['bbox'][1]))
    if not infos or not infos[0]['xref']: return None
    i = infos[0]; x = doc.extract_image(i['xref'])
    return x['image'], x['ext'], pymupdf.Rect(i['bbox']), x['width']


def main(pdf, saida, cod='por', ampliar=2.0, grafia_antiga=False, limite=0):
    idioma.definir(codigo=cod); idioma.GRAFIA_ANTIGA = grafia_antiga
    doc = pymupdf.open(pdf); ws = _o.cpu_count() or 2
    imgs = [imagem_da_pagina(doc, pg) for pg in doc]
    with tempfile.TemporaryDirectory() as td:
        normais, brancas = [], []
        for k, im in enumerate(imgs):
            if im is None: continue
            g = cv2.imdecode(np.frombuffer(im[0], np.uint8), cv2.IMREAD_GRAYSCALE)
            if ampliar != 1: g = cv2.resize(g, None, fx=ampliar, fy=ampliar, interpolation=cv2.INTER_CUBIC)
            a, b = Path(td) / f'{k:04d}.png', Path(td) / f'{k:04d}b.png'
            cv2.imwrite(str(a), g); cv2.imwrite(str(b), fundo_branco(g)); normais.append(a); brancas.append(b)
        with ThreadPoolExecutor(max(1, ws - 1)) as ex:
            ln = dict(ex.map(O.pagina, normais)); lb = dict(ex.map(O.pagina, brancas))
    lex = T.Lexico(set())
    ruins = lambda pg: sum(1 for w in pg['words'] if (n := T.nucleo(w['t'])) and len(n) > 1 and n.isalpha() and not lex.ok(n))
    palavras, trocadas, ilegiveis = {}, 0, []
    for a in normais:
        k = int(a.stem); pn, pb = ln[a.name], lb[a.stem + 'b.png']
        if ruins(pb) < ruins(pn): juntar_marcas(pb, pn, lex); pn = pb; trocadas += 1
        n = sum(1 for w in pn['words'] if (x := T.nucleo(w['t'])) and len(x) > 1 and x.isalpha())
        if limite and n >= 10 and ruins(pn) / n > limite: ilegiveis.append(k + 1); continue   # lixo só atrapalha a busca
        palavras[k] = pn['words']
    print(f'leitura com papel a branco escolhida em {trocadas} de {len(normais)} páginas')
    if ilegiveis: print(f'{len(ilegiveis)} páginas ilegíveis para o OCR, deixadas sem texto:', ilegiveis)
    out = pymupdf.open(); trocas = []
    for k, (pg, im) in enumerate(zip(doc, imgs)):
        np_ = out.new_page(width=pg.rect.width, height=pg.rect.height)
        if im is None: continue
        dados, ext, caixa, wpx = im
        np_.insert_image(caixa, stream=dados)
        if k in palavras:
            esc = caixa.width / (wpx * ampliar)
            camada_texto(np_, [dict(w, x=w['x'] + caixa.x0 / esc, y=w['y'] + caixa.y0 / esc) for w in palavras[k]], esc, lex, trocas)
    out.set_toc(doc.get_toc()); out.set_metadata({k: v for k, v in doc.metadata.items() if k in ('title', 'author', 'subject', 'keywords')})
    out.subset_fonts(); unicode_certo(out); out.save(saida, garbage=3, deflate=True)
    print(f'{Path(saida).name}: {len(out)} páginas, {_o.path.getsize(saida) / 1e6:.1f} MB; {len(trocas)} leituras corrigidas:',
          collections.Counter(f'{a} -> {b}' for a, b in trocas).most_common(20))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('pdf'); ap.add_argument('saida')
    ap.add_argument('--idioma', default='por'); ap.add_argument('--ampliar', type=float, default=2.0)
    ap.add_argument('--grafia-antiga', action='store_true')
    ap.add_argument('--sem-lixo', type=float, default=0, help='deixa sem texto a página com mais que esta fração de palavras fora do léxico')
    a = ap.parse_args()
    main(a.pdf, a.saida, a.idioma, a.ampliar, a.grafia_antiga, a.sem_lixo)
