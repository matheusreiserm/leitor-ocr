"""Protótipo: limpeza guiada pela geometria das linhas (vinda do OCR) e pelo léxico.

Uso: py -I -X utf8 limpar_por_moldes.py <pasta_entrada> <pasta_saida>
Para cada página: OCR (Tesseract, como ocr.py) -> linhas e palavras -> cada peça de tinta pequena é julgada:
  - entre linhas (fora de qualquer faixa de linha): sujeira;
  - acima da altura de x, sobre uma palavra: fica só o número de marcas que a palavra (corrigida pelo léxico)
    pede, na letra que as pede (pingo do i/j, acentos, til, trema); o resto é sujeira;
  - abaixo da linha de base, sob letra sem perna: sujeira.
Nunca acrescenta tinta. Grava também <saida>/apagado/<pagina>.png (em vermelho o que saiu).
"""
import sys, os, json, re, collections, unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
FER = Path(__file__).resolve().parent.parent / 'ferramentas_reconstrucao'; sys.path.insert(0, str(FER))  # repositório reconstrucao-livros, ao lado
import cv2, numpy as np
import ocr as O, triagem as T
import idioma; idioma.definir(codigo=os.environ.get('LEITOR_IDIOMA', 'spa'))   # 'por' para livros em português

MARCAS_ACIMA = {**{c: 1 for c in 'iíjáéóúñâêôãõàÁÉÍÓÚÑÂÊÔÃÕÀ'}, 'ü': 2, 'Ü': 2}
COM_PERNA = set('gjpqyçÇQ')
AMBIGUAS_RAZAO = 20   # "esta/está", "el/él": se as duas formas são comuns, não se mexe nas marcas da palavra
# Terminações que nunca levam acento no -o final: gerúndio, particípio ("preguntandó", "parecidó") e pronome
# colado ao infinitivo ("taparló"); "modulo" termina em -lo sem ser pronome, por isso o -r antes.
NUNCA_O_ACENTUADO = re.compile(r'(ndo|ido|ado|r(lo|la|le|los|las|les))$')


def acento_verbal_possivel(lido, certo):
    """O -ó final lido pode ser pretérito ("aleteó", "moduló") que o léxico, feito de legendas, não traz:
    a forma sem acento ("aleteo", "módulo") é outra palavra, não a correção. Em dúvida, não se mexe nas marcas."""
    return lido.endswith('ó') and not certo.endswith('ó') and not NUNCA_O_ACENTUADO.search(sem_acento_mantem_n(lido))
ONDE_MARCA = {}       # letra -> (dx, dy) da marca em relação ao corpo da letra e à linha de base, tirado dos moldes


def carregar_moldes(f):
    """Dos moldes (moldes.py), onde fica a marca de cada letra: centro da tinta acima da altura de x
    menos o centro do corpo da letra (dx), e altura em relação à linha de base (dy)."""
    import pickle
    M = pickle.load(open(f, 'rb')); A, xh = M['A'], M['xh']
    for ch, d in M['moldes'].items():
        if ch not in MARCAS_ACIMA: continue
        P = d['P'] * (d['P'] > 0.3); ys, xs = np.mgrid[0:P.shape[0], 0:P.shape[1]]
        marca, corpo = ys < A - 0.85 * xh, ys >= A - 1.0 * xh
        if P[marca].sum() == 0 or P[corpo].sum() == 0: continue
        cx = lambda z: (P[z] * xs[z]).sum() / P[z].sum()
        ONDE_MARCA[ch] = (float(cx(marca) - cx(corpo)), float((P[marca] * ys[marca]).sum() / P[marca].sum() - A))


def sem_acento_mantem_n(s):
    out = []
    for c in s:
        if c in 'ñÑ': out.append(c); continue
        out.append(''.join(x for x in unicodedata.normalize('NFD', c) if unicodedata.category(x) != 'Mn'))
    return ''.join(out)


class Formas:
    """Para cada forma sem acento, as grafias do léxico (com a frequência)."""
    def __init__(self, lex):
        self.lex = lex; self.por = collections.defaultdict(list)
        for w, n in lex.f.items():
            if n >= 3 and w.isalpha(): self.por[sem_acento_mantem_n(w)].append((n, w))
        for v in self.por.values(): v.sort(reverse=True)

    def validar(self, t):
        """Grafia certa da palavra lida, ou None se não dá para ter certeza das marcas."""
        n = T.nucleo(t)
        if len(n) < 2 or not n.isalpha(): return None
        low = n.lower(); chave = sem_acento_mantem_n(low)
        if idioma.grafia_antiga(low): return None   # "idéia", "vôo": grafia anterior a 2009, que o léxico não traz
        # português: pares que só o acento separa ("pôde/pode", "nós/nos", "camelôs/camelos"); se a palavra lida
        # existe no léxico, a marca dela não é tocada, por mais frequente que seja a outra forma
        if idioma.ATUAL == 'por' and low in self.lex.f: return n
        cands = self.por.get(chave, [])
        if not cands and 'ñ' in chave:   # til que o léxico não aceita ("eñañitos"): sujeira lida como til
            cands = self.por.get(chave.replace('ñ', 'n'), [])
        if not cands:
            return n if low in self.lex.livro else None
        if len(cands) > 1 and cands[1][0] * AMBIGUAS_RAZAO > cands[0][0]: return None   # ambígua
        best = cands[0][1]
        if idioma.ATUAL == 'spa' and acento_verbal_possivel(low, best): return None
        return ''.join(b.upper() if a.isupper() else b for a, b in zip(n, best)) if len(best) == len(n) else None


def limpar(gray, pg, formas):
    H, W = gray.shape
    words, chars = pg['words'], pg['chars']
    alt = [c[4] - c[2] for c in chars if c[0] in 'aceomnrsuvxz']
    if not alt: return gray.copy(), np.zeros_like(gray, bool), {}
    xh = float(np.median(alt))
    linhas = T.linhas_da_pagina(pg)
    bl = T.base_das_linhas(pg, linhas, xh)       # base por índice de palavra
    bin_ = (gray < 200).astype(np.uint8)
    k, lab, st, cen = cv2.connectedComponentsWithStats(bin_, 8)
    pequena = (st[:, 4] < 0.25 * xh * xh) & (st[:, 3] < 0.7 * xh)
    pequena[0] = False
    apagar = set(); motivo = collections.Counter()

    # faixas das linhas (do alto dos acentos até o pé das pernas)
    faixas = []
    for L in linhas:
        bs = [bl[i] for i in L if i in bl]
        # linha em letra menor (cabeçalho, versalete, colofão) fica sem base medida: vale o pé das palavras,
        # senão o til e os acentos dela caem como "entre linhas" ("João" do cabeçalho virava "Joao")
        if not bs: bs = [words[i]['y'] + words[i]['h'] for i in L]
        b = float(np.median(bs)); x0 = min(words[i]['x'] for i in L); x1 = max(words[i]['x'] + words[i]['w'] for i in L)
        faixas.append((b - 2.1 * xh, b + 1.0 * xh, x0 - 1.5 * xh, x1 + 1.5 * xh))
    if faixas:
        for i in np.where(pequena)[0]:
            cx, cy = cen[i]
            if not any(y0 <= cy <= y1 and x0 <= cx <= x1 for y0, y1, x0, x1 in faixas):
                apagar.add(i); motivo['entre linhas'] += 1

    cx_ch = np.array([(c[1] + c[3]) / 2 for c in chars]); cy_ch = np.array([(c[2] + c[4]) / 2 for c in chars])
    pequenas = np.where(pequena)[0]
    julgadas = []      # (v, cs, acima, abaixo, fica) das palavras validadas
    protegidas = set() # marcas que alguma palavra pede, ou sobre palavra que não se pôde validar
    # 1ª passada: o que cada palavra pede. As linhas deste livro são próximas: a zona abaixo de uma linha
    # cobre a zona dos acentos da seguinte, então a marca pedida por uma linha nunca é apagada pela outra.
    # pedaços de palavra hifenizada no fim da linha e o começo da linha seguinte: não são palavras ("dú-" + "vida")
    pedaco = set()
    for k_, L in enumerate(linhas):
        if L and words[L[-1]]['t'].endswith('-'):
            pedaco.add(L[-1])
            if k_ + 1 < len(linhas) and linhas[k_ + 1]: pedaco.add(linhas[k_ + 1][0])
    for wi, w in enumerate(words):
        if wi not in bl: continue
        b = bl[wi]
        acima_caixa = [i for i in pequenas if w['x'] <= cen[i][0] <= w['x'] + w['w']
                       and st[i][1] + st[i][3] <= b - 0.85 * xh and st[i][1] >= b - 2.3 * xh]
        v = None if wi in pedaco else formas.validar(w['t'])
        # posição real de cada letra: os corpos de tinta que atravessam a faixa da altura de x, em ordem
        corpos = sorted((i for i in range(1, k) if not pequena[i] and w['x'] - 2 <= cen[i][0] <= w['x'] + w['w'] + 2
                         and st[i][1] <= b - 0.6 * xh and st[i][1] + st[i][3] >= b - 0.2 * xh and st[i][3] >= 0.5 * xh),
                        key=lambda i: st[i][0])
        if v is None or len(corpos) != len(v):              # sem certeza da palavra ou da posição das letras
            protegidas.update(acima_caixa); continue
        cs = [[ch, st[i][0], 0, st[i][0] + st[i][2], 0, i] for ch, i in zip(v, corpos)]
        lx0, lx1 = cs[0][1], cs[-1][3]
        # peças pequenas sobre a palavra, entre as letras (aspas nas pontas ficam de fora)
        perto = [i for i in pequenas if lx0 <= cen[i][0] <= lx1]
        acima = [i for i in perto if st[i][1] + st[i][3] <= b - 0.85 * xh and st[i][1] >= b - 2.3 * xh]
        abaixo = [i for i in perto if st[i][1] >= b + 0.12 * xh and st[i][1] <= b + 1.0 * xh]
        # acima: cada letra que leva marca fica com as k peças mais perto de onde o molde da letra põe a marca
        fica = set()
        for ch, c in zip(v, cs):
            kk = MARCAS_ACIMA.get(ch, 0)
            if not kk: continue
            dx, dy = ONDE_MARCA.get(ch, (0.0, -1.35 * xh))
            ex, ey = cen[c[5]][0] + dx, b + dy
            cand = sorted((np.hypot(cen[i][0] - ex, cen[i][1] - ey), i) for i in acima if i not in fica)
            fica.update(i for d, i in cand[:kk] if d <= 0.7 * xh)
        protegidas.update(fica)
        julgadas.append((v, cs, acima, abaixo, fica))
    # Invariante: marca protegida nunca é apagada, nem se antes foi tomada por sujeira "entre linhas"
    # (a faixa da linha começa a 2,1 alturas de x da base; a zona dos acentos vai até 2,3).
    pendentes = apagar & protegidas
    apagar -= pendentes; motivo['entre linhas'] -= len(pendentes)
    # 2ª passada: o que sobra é sujeira
    for v, cs, acima, abaixo, fica in julgadas:
        for i in acima:
            if i not in fica and i not in protegidas and i not in apagar: apagar.add(i); motivo['acima da letra'] += 1
        # abaixo: só sob letra sem perna
        for i in abaixo:
            if i in protegidas or i in apagar: continue
            sob = [ch for ch, c in zip(v, cs) if c[1] - 2 <= cen[i][0] <= c[3] + 2]
            if sob and not any(ch in COM_PERNA for ch in sob):
                apagar.add(i); motivo['abaixo da letra'] += 1

    apagar -= protegidas   # última salvaguarda antes de apagar pixels
    m = np.isin(lab, list(apagar)) if apagar else np.zeros_like(gray, bool)
    out = gray.copy(); out[m] = 255
    return out, m, motivo


def pagina(args):
    f, saida, formas = args
    nome, pg = O.pagina(f)
    gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    out, m, motivo = limpar(gray, pg, formas)
    cv2.imwrite(str(saida / f.name), out)
    viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR); viz[m] = (0, 0, 255)
    cv2.imwrite(str(saida / 'apagado' / f.name), viz)
    return f.name, motivo


def main(entrada, saida):
    entrada, saida = Path(entrada), Path(saida); (saida / 'apagado').mkdir(parents=True, exist_ok=True)
    formas = Formas(T.Lexico(set()))
    carregar_moldes(Path.cwd() / '_moldes.pkl')   # gerado por moldes.py na pasta de trabalho
    print('onde fica a marca:', {c: (round(a, 1), round(b, 1)) for c, (a, b) in ONDE_MARCA.items()})
    fs = sorted(entrada.glob('*.png'))
    tot = collections.Counter()
    with ThreadPoolExecutor(max(1, (O.os.cpu_count() or 2) - 1)) as ex:
        for nome, motivo in ex.map(pagina, [(f, saida, formas) for f in fs]):
            tot.update(motivo); print(nome, dict(motivo), flush=True)
    print('total apagado:', dict(tot))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
