"""PDF pesquisável a partir de uma pasta de páginas: a imagem como está e o texto do OCR invisível por baixo.

Uso: py ocr_pdf.py "<pasta de imagens>" "<saida.pdf>" "<pasta do livro>" ["<pasta para o OCR>"]
Com a quarta pasta (ex.: paginas/, as mesmas páginas sem a textura de papel), o OCR lê dela, na mesma ordem e
no mesmo tamanho: a textura apaga acentos finos ("podía" -> "podia"), a página limpa não.
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
        textos = list(ex.map(lambda p: texto_invisivel(p[1], branquear=False) if com_texto(p[0]) else None, zip(fs, lidas)))   # página já limpa: sem branquear (perde acentos)
        palavras = dict(ex.map(O.pagina, [g for f, g in zip(fs, fo) if com_texto(f)]))
    larg = imgs[1].shape[1]; dpi = larg / (14 / 2.54)      # mesma escala do pdfs_livro.py: página de 14 cm
    out = pymupdf.open()
    for f, im, txt in zip(fs, imgs, textos):
        h, w = im.shape[:2]; pg = out.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        pg.insert_image(pg.rect, filename=str(f))
        if txt: pg.show_pdf_page(pg.rect, pymupdf.open('pdf', txt), 0)
    out.save(saida, garbage=3, deflate=True)
    json.dump(palavras, open(saida + '.ocr.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'{saida}: {len(fs)} páginas, {_o.path.getsize(saida) / 1e6:.1f} MB')


if __name__ == '__main__':
    main(*sys.argv[1:5])
