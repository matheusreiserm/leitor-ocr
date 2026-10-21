"""OCR (Tesseract e indra) de cada variante e comparação com o gabarito do Surya."""
import sys, json, re, difflib, subprocess, collections, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
FER = Path(__file__).resolve().parent.parent / 'ferramentas_reconstrucao'; sys.path.insert(0, str(FER))  # repositório reconstrucao-livros, ao lado
import ocr as O, triagem as T
AQ = Path.cwd()   # pasta de trabalho: variantes, gabarito_surya/ e _indra/
INDRA = shutil.which('indra-ocr') or str(Path(sys.executable).parent / 'Scripts' / 'indra-ocr.exe')
ESP = set('abcdefghijklmnopqrstuvwxyzáéíóúüñ')

def palavras(s):
    s = re.sub(r'<[^>]+>', ' ', s); s = re.sub(r'(\w)-\s+(\w)', r'\1\2', s)   # junta hifenização de fim de linha
    return re.findall(r'[^\W\d_]+', s.lower())

gab = {}
for f in (AQ / 'gabarito_surya').rglob('results.json'):
    for nome, pags in json.load(open(f, encoding='utf-8')).items():
        gab[nome] = palavras(' '.join(b.get('html', '') for b in pags[0]['blocks']))

def wer(ref, hyp):
    e = 0
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if op != 'equal': e += max(a1 - a0, b1 - b0)
    return e

lex = T.Lexico(set())
def medir(textos):
    n = fora = estr = err = nref = 0
    for nome, txt in textos.items():
        ws = palavras(txt)
        n += len(ws); fora += sum(1 for w in ws if len(w) > 1 and not lex.ok(w)); estr += sum(1 for w in ws if set(w) - ESP)
        r = gab.get(Path(nome).stem.split('.')[0])
        if r: err += wer(r, ws); nref += len(r)
    return f'{100*fora/n:5.2f}% fora do léxico | {estr:3} letras estranhas | WER vs Surya {100*err/max(nref,1):5.2f}%'

def tess(pasta):
    fs = sorted((AQ / pasta).glob('*.png'))
    with ThreadPoolExecutor(11) as ex:
        return {Path(n).stem: ' '.join(w['t'] for w in pg['words']) for n, pg in ex.map(O.pagina, fs)}

def indra(pasta):
    out = AQ / '_indra' / pasta; out.mkdir(parents=True, exist_ok=True)
    fs = [str(f) for f in sorted((AQ / pasta).glob('*.png'))]
    subprocess.run([INDRA, *fs, '--out', str(out)], check=True, capture_output=True)
    return {f.stem: f.read_text(encoding='utf-8') for f in out.glob('*.md')}

print(f'gabarito: {len(gab)} páginas, {sum(map(len, gab.values()))} palavras')
for p in sys.argv[1:]:
    print(f'{p:16} Tesseract: {medir(tess(p))}', flush=True)
    print(f'{"":16} indra:     {medir(indra(p))}', flush=True)
