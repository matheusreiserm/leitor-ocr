"""OCR (Tesseract e indra) de cada variante e comparação com o gabarito do Surya."""
import sys, os, json, re, subprocess, collections, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
FER = Path(__file__).resolve().parent.parent / 'ferramentas_reconstrucao'; sys.path.insert(0, str(FER))  # repositório reconstrucao-livros, ao lado
import ocr as O, triagem as T
import idioma; idioma.definir(codigo=os.environ.get('LEITOR_IDIOMA', 'spa'))   # 'por' para livros em português
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
    """Distância de edição em palavras (substituições + exclusões + inserções), o mínimo possível."""
    ant = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        atual = [i]
        for j, b in enumerate(hyp, 1):
            atual.append(min(ant[j] + 1, atual[j - 1] + 1, ant[j - 1] + (a != b)))
        ant = atual
    return ant[-1]

lex = T.Lexico(set())
def medir(textos):
    n_lex = fora = estr = err = nref = casadas = 0
    for nome, txt in textos.items():
        ws = palavras(txt)
        n_lex += sum(1 for w in ws if len(w) > 1)   # mesma base do comparar.py: palavras de mais de uma letra
        fora += sum(1 for w in ws if len(w) > 1 and not lex.ok(w)); estr += sum(1 for w in ws if set(w) - ESP)
        r = gab.get(Path(nome).stem.split('.')[0])
        if r is not None: err += wer(r, ws); nref += len(r); casadas += 1
    if not nref: raise ValueError('nenhuma página da variante casa com o gabarito do Surya: confira os nomes')
    return (f'{100*fora/n_lex:5.2f}% fora do léxico | {estr:3} letras estranhas | WER vs Surya {100*err/nref:5.2f}%'
            f' | páginas {casadas}/{len(textos)}')

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
