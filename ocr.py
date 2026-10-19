"""OCR das páginas limpas (Tesseract, espanhol) no formato que o resto das ferramentas lê.

Uso: py ocr.py "<pasta do livro limpo>"
Grava <pasta>/ocr.json: {pagina: {words: [{t, conf, x, y, w, h, line, wn}], chars: [[c, x0, y0, x1, y1]], shape}}
O modelo de espanhol fica em ferramentas_reconstrucao/tessdata (tessdata_best).
"""
import sys, json, os, subprocess, tempfile, csv
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2

TESS = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA = Path(__file__).parent / 'tessdata'


def pagina(f):
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE); H, W = img.shape
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / 'p.png'; cv2.imwrite(str(src), img)  # caminho sem acento para o Tesseract
        env = dict(os.environ, OMP_THREAD_LIMIT='1')
        subprocess.run([TESS, str(src), str(Path(td) / 'p'), '--tessdata-dir', str(TESSDATA), '-l', 'spa',
                        '--psm', '3', 'tsv', 'makebox'], check=True, capture_output=True, env=env)
        words = []
        with open(Path(td) / 'p.tsv', encoding='utf-8', newline='') as fh:
            for r in csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE):
                t = (r['text'] or '').strip()
                if r['level'] != '5' or not t: continue
                words.append(dict(t=t, conf=float(r['conf']), x=int(r['left']), y=int(r['top']), w=int(r['width']), h=int(r['height']),
                                  line=[int(r['block_num']), int(r['par_num']), int(r['line_num'])], wn=int(r['word_num'])))
        chars = []
        for ln in open(Path(td) / 'p.box', encoding='utf-8'):
            p = ln.rstrip('\n').rsplit(' ', 5)
            if len(p) != 6 or not p[0].strip(): continue
            c, l, b, r, t = p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])
            chars.append([c, l, H - t, r, H - b])  # a caixa do Tesseract mede y de baixo para cima
    return f.name, dict(words=words, chars=chars, shape=[H, W])


def main(pasta):
    pasta = Path(pasta)
    fs = sorted((pasta / 'paginas').glob('*.png'))
    out = {}
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for k, (nome, d) in enumerate(ex.map(pagina, fs), 1):
            out[nome] = d
            if k % 10 == 0 or k == len(fs): print(f'{k}/{len(fs)} páginas', flush=True)
    json.dump(out, open(pasta / 'ocr.json', 'w', encoding='utf-8'), ensure_ascii=False)
    n = sum(len(d['words']) for d in out.values())
    baixa = sum(w['conf'] < 80 for d in out.values() for w in d['words'])
    print(f'{n} palavras, {baixa} com confiança abaixo de 80%. Gravado em {pasta / "ocr.json"}')


if __name__ == '__main__':
    main(sys.argv[1])
