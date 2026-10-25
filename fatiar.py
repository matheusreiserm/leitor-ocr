"""Livro em páginas duplas já legível: só corta cada folha em duas páginas, sem mexer na imagem, e põe por baixo
o texto do OCR (invisível), para buscar e selecionar.

Uso: py fatiar.py "<livro.pdf>" "<pasta do livro>" "<título>"
Lê <pasta>/reconstrucao/config.json: "idioma", "ignorar" (folhas ou metades fora do miolo) e "capas" (recortes
de capa, orelhas e contracapa, postos antes e depois do miolo, também sem tratamento).
Nada de endireitar, clarear, limpar ou realinhar: cada página é o recorte do escaneamento original, em cor.
Grava <pasta>/<título>.pdf e <pasta>/paginas_originais/ (as páginas recortadas, para conferência).
"""
import sys, json, subprocess, tempfile
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pymupdf
import idioma
from preparar import fundo_branco, lombada_vao
from capas import imagem_cor
from ocr import TESS, TESSDATA

DPI = 300


def texto_invisivel(img):
    """PDF de uma página só com o texto do Tesseract (sem imagem), no tamanho da imagem a 300 dpi."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'p.png'; cv2.imwrite(str(src), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        subprocess.run([TESS, str(src), str(Path(td) / 'p'), '--tessdata-dir', str(TESSDATA), '-l', idioma.tesseract(),
                        '--dpi', str(DPI), '-c', 'textonly_pdf=1', 'pdf'], check=True, capture_output=True,
                       env=dict(_o.environ, OMP_THREAD_LIMIT='1'))
        return (Path(td) / 'p.pdf').read_bytes()


def main(pdf, pasta, titulo):
    pasta = Path(pasta); idioma.definir(pasta)
    cfg = json.load(open(pasta / 'reconstrucao' / 'config.json', encoding='utf-8-sig'))
    ignorar = {(int(f), l) for f, l in cfg.get('ignorar', [])}
    doc = pymupdf.open(pdf)
    paginas = []
    for i in range(len(doc)):
        if (i + 1, '*') in ignorar: continue
        c = imagem_cor(doc, i)
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        L = lombada_vao(np.where(fundo_branco(g) < 180, 0, 255).astype(np.uint8))   # só para achar onde cortar
        for lado, m in (('esq', c[:, :L]), ('dir', c[:, L:])):
            if (i + 1, lado) not in ignorar: paginas.append((f'{i + 1:02d}_{lado}', m))
    capas = cfg.get('capas', {})
    def recortes(lugar):
        return [(f'{lugar}_{k}_{it["nome"]}', imagem_cor(doc, it['folha'] - 1)[it['caixa'][1]:it['caixa'][3], it['caixa'][0]:it['caixa'][2]])
                for k, it in enumerate(capas.get(lugar, []), 1)]
    seq = recortes('antes') + paginas + recortes('depois')
    po = pasta / 'paginas_originais'; po.mkdir(exist_ok=True)
    for f in po.glob('*.jpg'): f.unlink()
    with ThreadPoolExecutor(max(1, (_o.cpu_count() or 2) - 1)) as ex:
        textos = list(ex.map(lambda nm: None if 'capa' in nm[0] or 'orelha' in nm[0] else texto_invisivel(nm[1]), seq))
    out = pymupdf.open()
    for k, ((nome, img), txt) in enumerate(zip(seq, textos), 1):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (po / f'{k:03d}_{nome}.jpg').write_bytes(buf.tobytes())
        h, w = img.shape[:2]; pg = out.new_page(width=w * 72 / DPI, height=h * 72 / DPI)
        pg.insert_image(pg.rect, stream=buf.tobytes())
        if txt:
            t = pymupdf.open('pdf', txt); pg.show_pdf_page(pg.rect, t, 0)
    dst = pasta / f'{titulo}.pdf'; out.save(str(dst), garbage=3, deflate=True)
    print(f'{dst.name}: {len(seq)} páginas ({len(paginas)} do miolo), {dst.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main(*sys.argv[1:4])
