"""Protótipo, 2ª camada: moldes de letra do próprio livro para tirar sujeira GRUDADA nas letras.

Uso: py -I -X utf8 moldes.py <pasta_livro_para_moldes> <pasta_entrada> <pasta_saida>
1. OCR de todas as páginas da pasta do livro; das palavras de confiança alta que o léxico confirma, cada letra
   (peça de tinta, com acento/pingo) vira amostra. Molde = média das amostras alinhadas pelo centro
   (probabilidade de tinta em cada pixel). A sujeira muda de lugar a cada cópia e some na média.
2. Em cada página de entrada: cada letra de palavra validada recebe o molde da letra (melhor encaixe);
   a tinta que o molde não explica, em manchas, é apagada. Só se a letra encaixa bem no molde
   (itálico, ligaduras e letras quebradas ficam como estão) e nunca mais de 30% da tinta da letra.
"""
import sys, pickle, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
AQ = Path(__file__).parent; sys.path.insert(0, str(AQ))
import limpar_por_moldes as L1
from limpar_por_moldes import O, T
import cv2, numpy as np

PAD = 4; MIN_AMOSTRAS = 15
BASE = set('acemnorsuvwxzáéíóúñ')   # letras apoiadas na linha de base, sem perna


def pecas(gray, w, t):
    """Como glifos._por_pecas, mas devolve (letra, y0, x0, máscara) em coordenadas da página."""
    X, Y, W, H = w['x'], w['y'], w['w'], w['h']
    p = int(H * 0.5); Y0, Y1 = max(0, Y - p), min(gray.shape[0], Y + H + p)
    reg = (gray[Y0:Y1, X:X + W] < 200).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(reg, 8)
    # peças que tocam o alto ou o pé da janela são pernas/hastes das linhas vizinhas, não desta palavra
    gs = sorted(([i] for i in range(1, k) if st[i][4] >= 6 and st[i][1] > 0 and st[i][1] + st[i][3] < reg.shape[0]),
                key=lambda g: st[g[0]][0])
    juntas = []
    for g in gs:
        x0, x1 = st[g[0]][0], st[g[0]][0] + st[g[0]][2]
        if juntas:
            a0 = min(st[i][0] for i in juntas[-1]); a1 = max(st[i][0] + st[i][2] for i in juntas[-1])
            if min(x1, a1) - max(x0, a0) > 0.5 * min(x1 - x0, a1 - a0): juntas[-1] += g; continue
        juntas.append(g)
    if len(juntas) != len(t): return [], None
    base = [Y0 + max(st[i][1] + st[i][3] for i in g) for c, g in zip(t, juntas) if c in BASE]
    if not base: return [], None
    out = []
    for c, g in zip(t, juntas):
        m = np.isin(lab, g); ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out.append((c, Y0 + y0, X + x0, m[y0:y1, x0:x1]))
    return out, int(np.median(base))


def ocr_pasta(pasta):
    fs = sorted(Path(pasta).glob('*.png'))
    with ThreadPoolExecutor(11) as ex:
        return list(zip(fs, (pg for _, pg in ex.map(O.pagina, fs))))


def construir(pasta, formas):
    am = collections.defaultdict(list); alt = []
    for f, pg in ocr_pasta(pasta):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        for w in pg['words']:
            t = w['t']
            if w['conf'] < 92 or not t.isalpha() or formas.validar(t) != t: continue
            ps, bl = pecas(gray, w, t)
            for c, y0, x0, m in ps:
                am[c].append((y0 - bl, m))
                if c in 'acemnorsuvxz': alt.append(m.shape[0])
    xh = float(np.median(alt)); A, D = int(round(2.1 * xh)), int(round(1.0 * xh))
    moldes = {}
    for c, L in am.items():
        if len(L) < MIN_AMOSTRAS: continue
        mh, mw, md = (np.median([v for v in col]) for col in zip(*[(m.shape[0], m.shape[1], dy) for dy, m in L]))
        L = [(dy, m) for dy, m in L if abs(m.shape[0] - mh) <= 0.1 * mh + 1 and abs(m.shape[1] - mw) <= 0.1 * mw + 1 and abs(dy - md) <= 3]
        if len(L) < MIN_AMOSTRAS: continue
        Wc = int(mw) + 2 * PAD; acc = np.zeros((A + D, Wc), np.float32)
        for dy, m in L:
            y = A + dy; x = (Wc - m.shape[1]) // 2
            if y < 0 or y + m.shape[0] > A + D or x < 0: continue
            acc[y:y + m.shape[0], x:x + m.shape[1]] += m
        moldes[c] = dict(P=acc / len(L), n=len(L))
    return dict(moldes=moldes, A=A, xh=xh)


def encaixar(m, y0, x0, bl, P, A):
    """Põe o molde na posição que a linha de base indica e procura o melhor ajuste fino (±3 px, ±2 px).
    Devolve (cobertura do núcleo do molde pela tinta, fração da tinta explicada, tinta não explicada)."""
    nuc = P > 0.5; perm = cv2.dilate((P > 0.08).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    Hm, Wm = P.shape; h, w = m.shape
    my0 = bl - A; mx0 = x0 + w // 2 - Wm // 2              # posição nominal do molde na página
    melhor = None
    for dy in range(-2, 3):
        for dx in range(-3, 4):
            oy, ox = (my0 + dy) - y0, (mx0 + dx) - x0         # deslocamento do molde em relação à peça
            ys0, ys1 = max(0, oy), min(h, oy + Hm); xs0, xs1 = max(0, ox), min(w, ox + Wm)
            if ys0 >= ys1 or xs0 >= xs1: continue
            sc = (m[ys0:ys1, xs0:xs1] & nuc[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]).sum()
            if melhor is None or sc > melhor[0]: melhor = (sc, oy, ox)
    if melhor is None: return 0, 0, None
    sc, oy, ox = melhor
    zona = np.zeros((h, w), bool)
    ys0, ys1 = max(0, oy), min(h, oy + Hm); xs0, xs1 = max(0, ox), min(w, ox + Wm)
    zona[ys0:ys1, xs0:xs1] = perm[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    return sc / max(nuc.sum(), 1), (m & zona).sum() / max(m.sum(), 1), m & ~zona


def continua_traco(s, resto):
    """A mancha prolonga uma haste (ponta de h, b, l, pé de q)? Olha a tinta da letra logo abaixo
    (ou acima) dela: se a mancha tem a largura dessa haste e está alinhada com ela, é traço, não sujeira."""
    x, y, w, h = s[0], s[1], s[2], s[3]
    for faixa in (resto[y + h:y + h + 3], resto[max(0, y - 3):y]):
        cols = np.where(faixa.any(0))[0]
        if not len(cols): continue
        a0, a1 = cols.min(), cols.max() + 1
        sob = min(x + w, a1) - max(x, a0)
        if sob >= 0.6 * w and w <= 1.5 * (a1 - a0): return True
    return False


def aplicar(gray, pg, formas, M):
    moldes, A = M['moldes'], M['A']
    out = gray.copy(); apag = np.zeros(gray.shape, bool); n = 0
    for w in pg['words']:
        v = formas.validar(w['t'])
        if not v or T.nucleo(w['t']) != w['t'] or len(v) != len(w['t']): continue
        ps, bl = pecas(gray, w, v)
        for c, y0, x0, m in ps:
            if c not in moldes: continue
            cob, expl, exc = encaixar(m, y0, x0, bl, moldes[c]['P'], A)
            if cob < 0.85 or expl < 0.7 or exc is None or not exc.any(): continue
            k, lab, st, _ = cv2.connectedComponentsWithStats(exc.astype(np.uint8), 8)
            resto = m & ~exc
            tira = np.isin(lab, [i for i in range(1, k) if st[i][4] >= 6 and not continua_traco(st[i], resto)])
            if not tira.any() or tira.sum() > 0.3 * m.sum(): continue
            ys, xs = np.where(tira); out[y0 + ys, x0 + xs] = 255; apag[y0 + ys, x0 + xs] = True; n += 1
    return out, apag, n


def main(livro, entrada, saida):
    formas = L1.Formas(T.Lexico(set()))
    cache = Path.cwd() / '_moldes.pkl'
    if cache.exists(): M = pickle.load(open(cache, 'rb'))
    else: M = construir(livro, formas); pickle.dump(M, open(cache, 'wb'))
    moldes = M['moldes']
    print(f'{len(moldes)} moldes:', ' '.join(f'{c}:{d["n"]}' for c, d in sorted(moldes.items(), key=lambda x: -x[1]['n'])), flush=True)
    saida = Path(saida); (saida / 'apagado').mkdir(parents=True, exist_ok=True)
    mos = np.vstack([np.hstack([cv2.resize((255 - 255 * d['P']).astype(np.uint8), (40, 84)) for d in list(moldes.values())[i:i + 20]]
                               + [np.full((84, 40), 255, np.uint8)] * max(0, 20 - len(list(moldes.values())[i:i + 20])))
                     for i in range(0, len(moldes), 20)])
    cv2.imwrite(str(saida / '_moldes.png'), mos)
    tot = 0
    for f, pg in ocr_pasta(entrada):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        out, apag, n = aplicar(gray, pg, formas, M); tot += n
        cv2.imwrite(str(saida / f.name), out)
        viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR); viz[apag] = (0, 0, 255); cv2.imwrite(str(saida / 'apagado' / f.name), viz)
        print(f.name, n, 'letras limpas', flush=True)
    print('total', tot)


if __name__ == '__main__':
    main(*sys.argv[1:4])
