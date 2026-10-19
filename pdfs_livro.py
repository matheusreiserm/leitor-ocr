import sys, cv2, img2pdf, numpy as np
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, sys.argv[1]); import papel
O, titulo = Path(sys.argv[2]), sys.argv[3]; pp = O / 'paginas_papel'; pp.mkdir(exist_ok=True)
fs = sorted((O / 'paginas').glob('*.png'))
for n, f in enumerate(fs, 1):
    im = cv2.imread(str(f))
    r = im if 'capa' in f.stem else np.clip(im.astype(np.float32) / 255 * papel.textura(im.shape[0], im.shape[1], 3000 + n, 'esq' if n % 2 == 0 else 'dir'), 0, 255).astype(np.uint8)
    cv2.imwrite(str(pp / (f.stem + '.jpg')), r, [cv2.IMWRITE_JPEG_QUALITY, 86, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
w = cv2.imread(str(fs[1])).shape[1]; dpi = w / (14 / 2.54); lay = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
for pasta, suf in (('paginas', 'limpo'), ('paginas_papel', 'papel')):
    a = sorted(str(p) for p in (O / pasta).iterdir()); dst = O / f'{titulo} - {suf}.pdf'
    dst.write_bytes(img2pdf.convert(a, layout_fun=lay)); print(dst.name, len(a), f'{dst.stat().st_size/1e6:.1f} MB')
