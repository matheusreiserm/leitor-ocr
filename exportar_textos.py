"""Texto por página em .txt, para o avaliar_gabarito.py.

Uso:
  py -I -X utf8 exportar_textos.py ocr <pasta_de_paginas> <saida>          # <saida>/tesseract/*.txt e <saida>/indra/*.txt
  py -I -X utf8 exportar_textos.py surya <results.json> [...] --saida <dir> # gabarito do Surya, um .txt por página
Em todos, a hifenização de fim de linha é desfeita ("aprove-\\nchaba" -> "aprovechaba") e as marcas HTML do
Surya saem, para que referência e leitura sejam comparadas da mesma forma.
"""
import sys, json, re, html, shutil, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ferramentas_reconstrucao'))  # repositório reconstrucao-livros, ao lado
INDRA = shutil.which('indra-ocr') or str(Path(sys.executable).parent / 'Scripts' / 'indra-ocr.exe')


def normalizar(s):
    s = re.sub(r'</p>|<br\s*/?>', '\n\n', s)               # parágrafos do Surya
    s = html.unescape(re.sub(r'<[^>]+>', ' ', s))
    s = re.sub(r'(\w)[-¬]\s*\n\s*(\w)', r'\1\2', s)      # hífen no fim da linha
    s = re.sub(r'(\w)-\s+(\w)', r'\1\2', s)                # idem, quando a quebra já virou espaço
    s = re.sub(r'[ \t]+', ' ', s)
    return re.sub(r'\n\s*\n\s*', '\n\n', s).strip() + '\n'


def tesseract(paginas, saida):
    import ocr as O
    fs = sorted(Path(paginas).glob('*.png')); saida.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(11) as ex:
        for nome, pg in ex.map(O.pagina, fs):
            linhas = {}
            for w in pg['words']: linhas.setdefault(tuple(w['line']), []).append(w['t'])
            (saida / (Path(nome).stem + '.txt')).write_text(normalizar('\n'.join(' '.join(v) for v in linhas.values())), encoding='utf-8')


def indra(paginas, saida):
    fs = [str(f) for f in sorted(Path(paginas).glob('*.png'))]; tmp = saida.parent / '_indra_md'
    tmp.mkdir(parents=True, exist_ok=True); saida.mkdir(parents=True, exist_ok=True)
    subprocess.run([INDRA, *fs, '--out', str(tmp)], check=True, capture_output=True)
    for f in tmp.glob('*.md'): (saida / (f.stem + '.txt')).write_text(normalizar(f.read_text(encoding='utf-8')), encoding='utf-8')
    shutil.rmtree(tmp)


def surya(jsons, saida):
    saida.mkdir(parents=True, exist_ok=True); n = 0
    for j in jsons:
        for nome, pags in json.load(open(j, encoding='utf-8')).items():
            txt = '\n'.join(b.get('html', '') for b in pags[0]['blocks'])
            (saida / (Path(nome).stem + '.txt')).write_text(normalizar(txt), encoding='utf-8'); n += 1
    print(n, 'páginas de gabarito')


if __name__ == '__main__':
    modo = sys.argv[1]
    if modo == 'ocr':
        pag, out = Path(sys.argv[2]), Path(sys.argv[3])
        tesseract(pag, out / 'tesseract'); indra(pag, out / 'indra')
        print(out, ':', len(list((out / 'tesseract').glob('*.txt'))), 'páginas')
    elif modo == 'surya':
        i = sys.argv.index('--saida'); surya([Path(p) for p in sys.argv[2:i]], Path(sys.argv[i + 1]))
    else:
        sys.exit(__doc__)
