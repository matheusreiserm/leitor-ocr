"""PDF pesquisável a partir de uma pasta de páginas: a imagem como está e o texto do OCR invisível por baixo.

Uso: py ocr_pdf.py "<pasta de imagens>" "<saida.pdf>" "<pasta do livro>" ["<pasta para o OCR>"]
Com a quarta pasta (ex.: paginas/, as mesmas páginas sem a textura de papel), o OCR lê dela, na mesma ordem e
no mesmo tamanho: a textura apaga acentos finos ("podía" -> "podia"), a página limpa não.
A camada de texto é escrita a partir das caixas de palavra do OCR, com as leituras erradas de ü e í
corrigidas pelo léxico ("lingiiistas" -> "lingüistas", "enriquecií" -> "enriquecí"); a imagem não muda.
O idioma vem de <pasta do livro>/reconstrucao/config.json (sem ele, espanhol). Páginas com "capa" ou "orelha" no
nome ficam sem texto. Grava também <saida>.ocr.json com as palavras lidas (para avaliar a leitura).
"""
import sys, json
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pymupdf
import idioma
from fatiar import texto_invisivel, DPI
import ocr as O
import re, triagem as T

# leituras erradas de letras que o OCR não reconhece bem; só valem se a palavra lida não existe e a corrigida existe
_VARIANTES = [(re.compile(r'g(?:iii|ii|i|ú|u)(?=[aeiouáéíóú])'), 'gü'), (re.compile(r'q(?:iii|ii|ú)(?=[aeiouáéíóú])'), 'qü'),
              (re.compile(r'ií'), 'í'), (re.compile(r'iií'), 'í')]


def corrigir(palavra, lex):
    m = re.match(r'^(\W*)(.*?)(\W*)$', palavra); a, w, z = m.groups()
    if not w.isalpha() or lex.ok(w): return palavra
    cands = {w}
    for rx, novo in _VARIANTES:
        cands |= {rx.sub(novo, c) for c in cands}
    boas = [c for c in cands if c != w and lex.ok(c)]
    return a + max(boas, key=lambda c: lex.f.get(c.lower(), 0)) + z if boas else palavra


def camada_texto(pg, palavras, escala, lex, trocas):
    """Texto invisível escrito a partir das caixas de palavra do OCR (escala: pt por pixel), já corrigido."""
    for w in palavras:
        t = corrigir(w['t'], lex)
        if t != w['t']: trocas.append((w['t'], t))
        x, y, ww, hh = (v * escala for v in (w['x'], w['y'], w['w'], w['h']))
        l1 = pymupdf.get_text_length(t, fontname='helv', fontsize=1) or 1
        fs = max(1.0, min(ww / l1, hh * 1.3))
        pg.insert_text((x, y + hh * 0.85), t + ' ', fontname='helv', fontsize=fs, render_mode=3)


def main(pasta_img, saida, livro, pasta_ocr=None):
    idioma.definir(livro)
    fs = sorted(f for f in Path(pasta_img).iterdir() if f.suffix.lower() in ('.jpg', '.png'))
    com_texto = lambda f: 'capa' not in f.stem and 'orelha' not in f.stem
    imgs = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in fs]
    fo = sorted(f for f in Path(pasta_ocr).iterdir() if f.suffix.lower() in ('.jpg', '.png')) if pasta_ocr else fs
    assert len(fo) == len(fs), 'as duas pastas não têm o mesmo número de páginas'
    lidas = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in fo]
    assert all(a.shape[:2] == b.shape[:2] for a, b in zip(imgs, lidas)), 'páginas de tamanhos diferentes'
    with ThreadPoolExecutor(max(1, (_o.cpu_count() or 2) - 1)) as ex:
        palavras = dict(ex.map(O.pagina, [g for f, g in zip(fs, fo) if com_texto(f)]))   # página limpa como está
    lex = T.Lexico(set()); trocas = []
    larg = imgs[1].shape[1]; dpi = larg / (14 / 2.54)      # mesma escala do pdfs_livro.py: página de 14 cm
    out = pymupdf.open()
    for f, g, im in zip(fs, fo, imgs):
        h, w = im.shape[:2]; pg = out.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        pg.insert_image(pg.rect, filename=str(f))
        if g.name in palavras: camada_texto(pg, palavras[g.name]['words'], 72 / dpi, lex, trocas)
    out.save(saida, garbage=3, deflate=True)
    json.dump(palavras, open(saida + '.ocr.json', 'w', encoding='utf-8'), ensure_ascii=False)
    import collections
    print(f'{saida}: {len(fs)} páginas, {_o.path.getsize(saida) / 1e6:.1f} MB; {len(trocas)} leituras corrigidas:',
          collections.Counter(f'{a} -> {b}' for a, b in trocas).most_common(30))


if __name__ == '__main__':
    main(*sys.argv[1:5])
