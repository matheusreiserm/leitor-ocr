"""PDFs do livro: "<título> - limpo.pdf" (páginas como estão) e "<título> - papel.pdf" (com textura de papel).

Uso: py pdfs_livro.py <pasta das ferramentas> "<pasta do livro limpo>" "<título>"
As imagens de <pasta>/capas/ (feitas pelo capas.py) entram antes (antes_*) e depois (depois_*) do miolo.
Com "papel": false em reconstrucao/config.json, só o limpo.
Capa, contracapa e orelhas ficam como no escaneamento, sem textura; a folha de rosto ("rosto") recebe a textura
como as páginas. Arquivo com "capa" no nome dentro de paginas/ também fica sem textura (livros antigos).
"""
import sys, json, cv2, img2pdf, numpy as np
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, sys.argv[1]); import papel
O, titulo = Path(sys.argv[2]), sys.argv[3]; pp = O / 'paginas_papel'
fc = O / 'reconstrucao' / 'config.json'
# "papel": false -> só o PDF limpo (livro cujo papel tem cor própria: a textura artificial não acrescenta nada)
com_papel = (json.load(open(fc, encoding='utf-8-sig')) if fc.exists() else {}).get('papel', True)
if com_papel:
    pp.mkdir(exist_ok=True)
    for f in pp.glob('*.jpg'): f.unlink()
fs = sorted((O / 'paginas').glob('*.png'))
ca = O / 'capas'
antes = sorted(ca.glob('antes_*.png')) if ca.exists() else []
depois = sorted(ca.glob('depois_*.png')) if ca.exists() else []
seq = antes + fs + depois
sem_textura = lambda f: 'capa' in f.stem or 'orelha' in f.stem
for n, f in enumerate(seq if com_papel else [], 1):
    im = cv2.imread(str(f))
    r = im if sem_textura(f) else np.clip(im.astype(np.float32) / 255 * papel.textura(im.shape[0], im.shape[1], 3000 + n, 'esq' if n % 2 == 0 else 'dir'), 0, 255).astype(np.uint8)
    cv2.imwrite(str(pp / f'{n:03d}_{f.stem}.jpg'), r, [cv2.IMWRITE_JPEG_QUALITY, 86, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
w = cv2.imread(str(fs[1])).shape[1]; dpi = w / (14 / 2.54); lay = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
for imgs, suf in [(seq, 'limpo')] + ([(sorted(pp.glob('*.jpg')), 'papel')] if com_papel else []):
    a = [str(p) for p in imgs]; dst = O / f'{titulo} - {suf}.pdf'
    dst.write_bytes(img2pdf.convert(a, layout_fun=lay)); print(dst.name, len(a), f'{dst.stat().st_size/1e6:.1f} MB')
