"""Texto do livro: a leitura do Surya, corrigida só onde os outros dois motores concordam e o léxico decide.

Uso: py -I -X utf8 texto_consenso.py <gabarito_surya_txt> <tesseract_txt> <indra_txt> <saida.md> [titulo]
Uma palavra do Surya é trocada quando: (1) não está no léxico; (2) Tesseract e indra leram, no mesmo
lugar (alinhamento por distância de edição), a mesma palavra; (3) essa palavra está no léxico. A pontuação
do Surya é mantida. Cada troca vai para <saida>.trocas.csv, para conferência.
"""
import sys, csv, re
from pathlib import Path
FER = Path(__file__).resolve().parent.parent / 'ferramentas_reconstrucao'; sys.path.insert(0, str(FER))  # repositório reconstrucao-livros, ao lado
import triagem as T

PAL = re.compile(r'[^\W\d_]+')


def alinhar(a, b):
    """Para cada posição de a, a palavra de b alinhada por substituição/igualdade (ou None)."""
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1): d[i][0] = i
    for j in range(1, n + 1): d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    par = [None] * m; i, j = m, n
    while i and j:
        if d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]): par[i - 1] = b[j - 1]; i -= 1; j -= 1
        elif d[i][j] == d[i - 1][j] + 1: i -= 1
        else: j -= 1
    return par


def main(surya, tess, indra, saida, titulo='La Vida Nueva'):
    surya, tess, indra, saida = Path(surya), Path(tess), Path(indra), Path(saida)
    lex = T.Lexico(set())
    partes, trocas = [f'# {titulo}\n'], []
    for f in sorted(surya.glob('*.txt')):
        txt = f.read_text(encoding='utf-8')
        toks = list(PAL.finditer(txt)); s = [t.group().lower() for t in toks]
        lt = [w.lower() for w in PAL.findall((tess / f.name).read_text(encoding='utf-8'))] if (tess / f.name).exists() else []
        li = [w.lower() for w in PAL.findall((indra / f.name).read_text(encoding='utf-8'))] if (indra / f.name).exists() else []
        pt, pi = alinhar(s, lt), alinhar(s, li)
        novo, ult = [], 0
        for k, t in enumerate(toks):
            w = t.group(); troca = None
            if len(w) > 1 and not lex.ok(w) and pt[k] and pt[k] == pi[k] and pt[k] != w.lower() and lex.ok(pt[k]):
                troca = pt[k].capitalize() if w[0].isupper() else pt[k]
                trocas.append(dict(pagina=f.stem, surya=w, consenso=troca, contexto=txt[max(0, t.start() - 40):t.end() + 40].replace('\n', ' ')))
            novo.append(txt[ult:t.start()] + (troca or w)); ult = t.end()
        novo.append(txt[ult:])
        partes.append(f'\n<!-- página {f.stem.split("_")[0]} -->\n\n' + ''.join(novo).strip() + '\n')
    saida.write_text(''.join(partes), encoding='utf-8')
    with open(saida.with_suffix('.trocas.csv'), 'w', newline='', encoding='utf-8-sig') as fp:
        wr = csv.DictWriter(fp, fieldnames=['pagina', 'surya', 'consenso', 'contexto']); wr.writeheader(); wr.writerows(trocas)
    print(f'{saida}: {len(trocas)} palavras do Surya trocadas pelo consenso Tesseract + indra + léxico')


if __name__ == '__main__':
    main(*sys.argv[1:6])
