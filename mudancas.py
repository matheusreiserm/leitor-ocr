"""Palavras cuja leitura (Tesseract) mudou entre duas variantes, casadas pela posição; recorta cada uma."""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ferramentas_reconstrucao'))  # repositório reconstrucao-livros, ao lado
import os, ocr as O, cv2, numpy as np
import idioma; idioma.definir(codigo=os.environ.get('LEITOR_IDIOMA', 'spa'))   # 'por' para livros em português
a, b = sys.argv[1], sys.argv[2]; out = Path('mudou'); out.mkdir(exist_ok=True)
fs = sorted(Path(a).glob('*.png'))
with ThreadPoolExecutor(11) as ex:
    A = dict(ex.map(O.pagina, fs)); B = dict(ex.map(O.pagina, [Path(b) / f.name for f in fs]))
n = 0
for nome in A:
    for w in A[nome]['words']:
        par = [w2 for w2 in B[nome]['words'] if abs(w2['x'] - w['x']) < 8 and abs(w2['y'] - w['y']) < 8 and abs(w2['w'] - w['w']) < 15]
        t2 = par[0]['t'] if par else None
        if t2 == w['t']: continue
        n += 1; x, y, ww, h = w['x'], w['y'], w['w'], w['h']
        # sem rótulo de "certa": achar a grafia em algum lugar da página do Surya não diz nada sobre esta ocorrência
        print(f'{nome} {w["t"]!r:18} -> {t2!r:18} | revisão visual: {out / f"{n:02d}_{Path(nome).stem}.png"}')
        va = cv2.imread(str(Path(a) / nome))[max(0, y - 25):y + h + 10, max(0, x - 10):x + ww + 10]
        vb = cv2.imread(str(Path(b) / 'apagado' / nome))[max(0, y - 25):y + h + 10, max(0, x - 10):x + ww + 10]
        cv2.imwrite(str(out / f'{n:02d}_{Path(nome).stem}.png'), cv2.resize(np.vstack([va, vb]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
print(n, 'mudanças')
