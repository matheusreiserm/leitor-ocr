"""Separa as páginas duplas de um PDF escaneado em preto e branco, limpa a sujeira e amplia.

Uso: py preparar.py "<livro.pdf>" "<pasta do livro limpo>" [escala]
Grava <pasta>/paginas/NNN_pagina.png (texto preto em fundo branco, bordas suavizadas).

Limpeza (feita na resolução do escaneamento, sobre as peças de tinta conectadas):
- bordas pretas, sombra da lombada e riscos que encostam na borda da página;
- sujeira fora da mancha de texto e entre as linhas;
- pontos soltos no meio da altura das letras (não são pontuação) e traços abaixo da linha de base.
A lombada fica sempre na borda interna da página, então a mancha mantém a posição do livro;
na vertical, as folhas são alinhadas pelo número de página.
"""
import sys, json
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2, numpy as np, pymupdf


def imagem(doc, i):
    x = doc.extract_image(doc[i].get_images()[0][0])
    return cv2.imdecode(np.frombuffer(x['image'], np.uint8), cv2.IMREAD_GRAYSCALE)


def lombada(g):
    W = g.shape[1]; a, b = int(W * 0.35), int(W * 0.65)
    col = cv2.blur((g < 100).astype(np.float32).mean(0)[None], (9, 1))[0]
    return a + int(np.argmax(col[a:b]))


def xh_livro(gs):
    hs = []
    for g in gs:
        k, _, st, _ = cv2.connectedComponentsWithStats((g < 128).astype(np.uint8), 8)
        h, w = st[1:, 3], st[1:, 2]
        hs += list(h[(h > 8) & (h < 40) & (w < 40) & (w > 5)])
    v, c = np.unique(hs, return_counts=True)
    return float(v[np.argmax(c)])  # altura mais comum = letras sem haste (a, e, o, n...)


def _pecas(g):
    k, lab, st, cen = cv2.connectedComponentsWithStats((g < 128).astype(np.uint8), 8)
    return k, lab, st, cen


def _segmentos(ids, x0, w, xh):
    """Agrupa peças vizinhas (espaço até 0,4 x): cada grupo é uma palavra, ou um amontoado de lascas."""
    segs, fim = [], -1e9
    for i in sorted(ids, key=lambda i: x0[i]):
        if x0[i] - fim > 0.4 * xh: segs.append([])
        segs[-1].append(i); fim = max(fim, x0[i] + w[i])
    return segs


def _bandas(st, cen, letra, H, xh):
    """Linhas de texto: faixas horizontais cobertas pelo miolo (altura de x) das letras."""
    x0, y0, w, h = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    prof = np.zeros(H + 1, int)
    for i in np.where(letra & (h <= 2.2 * xh) & (w >= 0.2 * xh))[0]:  # sem riscos finos e altos, que juntariam duas linhas
        prof[int(y0[i] + 0.25 * h[i]):int(y0[i] + 0.75 * h[i]) + 1] += 1
    linhas, ini = [], None
    for y in range(H + 1):
        on = prof[y] > 0
        if on and ini is None: ini = y
        if not on and ini is not None: linhas.append((ini, y)); ini = None
    li = np.where(letra)[0]
    bandas = []
    for t, b in linhas:
        ids = sorted((i for i in li if t <= cen[i][1] <= b), key=lambda i: x0[i])
        if not ids: continue
        xl = np.percentile([h[i] for i in ids], 30)  # altura de x desta linha (títulos têm letra maior)
        fino = {i: w[i] <= 0.35 * xh for i in ids}  # traço fino: "l", "i" ou lasca
        # separa a linha em blocos (espaços maiores que 2,5 x) e descarta os que não parecem texto:
        # lascas empilhadas perto da borda, marcas na margem
        blocos, fim = [], -1e9
        for i in ids:
            if x0[i] - fim > 2.5 * max(xh, xl): blocos.append([])
            blocos[-1].append(i); fim = max(fim, x0[i] + w[i])
        cheias = lambda bl_: sum(not fino[i] for i in bl_)
        principal = max(blocos, key=cheias)
        ids = []
        for bl_ in blocos:
            if not (bl_ is principal or cheias(bl_) >= 2): continue
            # amontoados só de traços finos (sem nenhuma letra cheia por perto) são lascas
            ids += [i for s in _segmentos(bl_, x0, w, xh) if cheias(s) for i in s]
        if not ids: continue
        xs = [i for i in ids if 0.8 * xl <= h[i] <= 1.25 * xl]  # letras sem haste: pé na linha de base
        if not xs: continue
        bandas.append(dict(ids=ids, n=len(ids), x0=int(x0[ids[0]]), x1=int(x0[ids[-1]] + w[ids[-1]]),
                           px=np.array([cen[i][0] for i in xs]), pb=np.array([y0[i] + h[i] for i in xs], float),
                           pt=np.array([y0[i] for i in xs], float)))
    for b in bandas: b['bl'], b['xt'] = float(np.median(b['pb'])), float(np.median(b['pt']))
    return bandas


def _local(b, x, k='pb'):
    """Linha de base (ou topo do x) na posição x, pelas letras vizinhas: aguenta inclinação e página curva."""
    o = np.argsort(np.abs(b['px'] - x))[:7]
    return float(np.median(b[k][o]))


def inclinacao(g, xh):
    """Ângulo das linhas (graus; positivo = descem para a direita). Perfil de projeção: o ângulo em que o pé
    das letras se concentra em linhas mais nítidas. Funciona mesmo com inclinação forte, quando as faixas
    horizontais das linhas se fundem."""
    k, lab, st, cen = _pecas(g)
    x0, y0, w, h = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    sel = (h >= 0.6 * xh) & (h <= 1.6 * xh) & (w <= 2 * xh) & (w >= 0.15 * xh); sel[0] = False
    if sel.sum() < 40: return 0.0
    px = (x0 + w / 2)[sel].astype(np.float64); py = (y0 + h)[sel].astype(np.float64)
    px -= px.mean()
    def nitidez(a):
        yy = py - px * np.tan(np.radians(a))
        hist = np.bincount(((yy - yy.min()) / 2).astype(int))
        return float((hist.astype(np.float64) ** 2).sum())
    grade = np.arange(-5, 5.001, 0.2)
    a = grade[int(np.argmax([nitidez(v) for v in grade]))]
    fina = np.arange(a - 0.2, a + 0.2001, 0.02)
    return float(fina[int(np.argmax([nitidez(v) for v in fina]))])


def endireitar(g, ang):
    if abs(ang) < 0.05: return g
    H, W = g.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    r = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    return np.where(r < 128, 0, 255).astype(np.uint8)


def _corridas(linha):
    v = np.concatenate([[0], linha.astype(np.int8), [0]])
    return int((np.diff(v) == 1).sum())


def _forma_n(lab, st, i):
    """A letra tem o formato de 'n' (arco fechado em cima, duas hastes embaixo)? Só sobre ela cabe um til."""
    x, y, w, h = st[i][:4]
    m = lab[y:y + h, x:x + w] == i
    return _corridas(m[int(h * 0.8)]) == 2 and _corridas(m[int(h * 0.25)]) == 1


def colunas_de_risco(st, cen, bandas, xh):
    """Risco vertical do escaneamento: traços finos alinhados numa mesma coluna ao longo de muitas linhas,
    que não apoiam o pé na linha de base (a haste de um "l" ou "i" apoia). Devolve as peças a apagar."""
    x0, y0, w, h = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    flut = []
    for i in range(1, len(st)):
        if w[i] > 0.4 * xh or h[i] < 0.3 * xh or h[i] > 8 * xh: continue
        cx, cy = cen[i]
        dono = [b for b in bandas if b['xt'] - 1.0 * xh <= cy <= b['bl'] + 0.6 * xh and b['x0'] - xh <= cx <= b['x1'] + xh]
        if dono:
            b = min(dono, key=lambda b: abs(cy - (b['bl'] + b['xt']) / 2)); bl = _local(b, cx)
            if abs(y0[i] + h[i] - bl) <= 0.35 * xh: continue  # haste de letra (o pé na linha; o topo pode vir fundido com acento ou risco)
        flut.append(i)
    if len(flut) < 5: return {}
    xs = np.array([cen[i][0] for i in flut]); ys = np.array([cen[i][1] for i in flut])
    colunas = [c for c in np.unique(np.round(xs / 3) * 3) if (np.abs(xs - c) <= 4).sum() >= 5 and np.ptp(ys[np.abs(xs - c) <= 4]) > 8 * xh]
    acao = {}
    if not colunas: return acao
    # na coluna do risco: peças finas soltas saem; a que atravessa a linha de base dentro de uma palavra
    # contém uma letra (o risco passa por cima dela): fica, aparada à altura que uma letra pode ter
    for i in range(1, len(st)):
        if w[i] > 0.6 * xh or h[i] < 0.3 * xh or not any(abs(cen[i][0] - c) <= 4 for c in colunas): continue
        cx, cy = cen[i]
        dono = [b for b in bandas if b['xt'] - 1.0 * xh <= cy <= b['bl'] + 0.6 * xh and b['x0'] - xh <= cx <= b['x1'] + xh]
        if dono:
            b = min(dono, key=lambda b: abs(cy - (b['bl'] + b['xt']) / 2)); bl = _local(b, cx)
            cruza = y0[i] < bl - 0.5 * xh and y0[i] + h[i] > bl - 0.2 * xh
            vizinhas = [j for j in b['ids'] if j != i and w[j] > 0.35 * xh and
                        (0 <= x0[i] - (x0[j] + w[j]) <= 0.4 * xh or 0 <= x0[j] - (x0[i] + w[i]) <= 0.4 * xh or (x0[j] < x0[i] + w[i] and x0[j] + w[j] > x0[i]))]
            if cruza and vizinhas:
                if y0[i] < bl - 1.75 * xh or y0[i] + h[i] > bl + 0.1 * xh: acao[i] = ('aparar', bl - 1.75 * xh, bl + 0.1 * xh)
                continue
            if abs(y0[i] + h[i] - bl) <= 0.35 * xh and vizinhas: continue  # haste de letra com o pé na linha
            # risco que desce e termina no acento de uma letra (o "í" de "había"): fica só a parte na altura do acento
            xt = _local(b, cx, 'pt'); pe = y0[i] + h[i]
            embaixo = [j for j in b['ids'] if j != i and x0[j] <= cx <= x0[j] + w[j] and 0 <= y0[j] - pe <= 0.5 * xh]
            if xt - 0.4 * xh <= pe <= xt + 0.4 * xh and embaixo:
                acao[i] = ('aparar', bl - 1.6 * xh, bl + 0.1 * xh); continue
        if i in flut and (h[i] >= 0.6 * xh or not any(b['xt'] - 0.3 * xh <= cy <= b['bl'] + 0.3 * xh for b in bandas)):
            acao[i] = ('apagar',)
    return acao


def analisar(g, xh):
    """Peças de tinta, linhas de texto e a mancha. Devolve dict com o necessário para limpar."""
    H, W = g.shape
    k, lab, st, cen = _pecas(g)
    x0, y0, w, h, a = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
    borda = (x0 <= 1) | (y0 <= 1) | (x0 + w >= W - 1) | (y0 + h >= H - 1)
    faixa = 0.025
    canto = (x0 + w < W * faixa) | (x0 > W * (1 - faixa)) | (y0 + h < H * faixa) | (y0 > H * (1 - faixa))
    letra = (h >= 0.6 * xh) & (h <= 3.2 * xh) & (w <= 3.5 * xh) & (w >= 0.15 * xh) & ~borda & ~canto
    letra[0] = False
    bandas = _bandas(st, cen, letra, H, xh)
    texto = [b for b in bandas if b['n'] >= 3 and b['x1'] - b['x0'] >= 3 * xh]  # linha de verdade (não só riscos)
    vazio = dict(lab=lab, manter=[], bandas=[], caixa=None, folio=None, ficou=1.0, nletras=0)
    if not texto: return vazio
    plenas = [b for b in texto if b['x1'] - b['x0'] >= 10 * xh] or texto
    # margens da mancha (o texto não é justificado: a margem direita é irregular).
    # Percentis, para que um risco que escapou numa linha não alargue a mancha.
    # à esquerda (alinhada) um percentil basta; à direita (irregular) vale a linha mais longa, senão a última
    # letra das linhas compridas é cortada (as lascas já saíram das linhas pela regra dos segmentos)
    p = 5 if len(plenas) >= 8 else 0
    bx0 = float(np.percentile([b['x0'] for b in plenas], p)); bx1 = float(max(b['x1'] for b in plenas))
    eh_texto = {id(b) for b in texto}
    bandas = [b for b in bandas if b['x1'] > bx0 - xh and b['x0'] < bx1 + xh and (id(b) in eh_texto or b['x1'] - b['x0'] < 4 * xh)]
    by0 = min(b['xt'] for b in bandas) - 1.6 * xh; by1 = max(b['bl'] for b in bandas) + 0.9 * xh
    E0, E1, T0, T1 = bx0 - 0.6 * xh, bx1 + 0.6 * xh, by0 - 0.8 * xh, by1 + 0.6 * xh
    # peças por linha (para medir o espaço até a vizinha)
    zona = {id(b): [] for b in bandas}
    for i in range(1, k):
        for b in bandas:
            if b['xt'] - 0.3 * xh <= cen[i][1] <= b['bl'] + 0.5 * xh: zona[id(b)].append(i); break
    riscos = colunas_de_risco(st, cen, bandas, xh)
    aparar = [(i, a[1], a[2]) for i, a in riscos.items() if a[0] == 'aparar']
    manter = []
    for i in range(1, k):
        if borda[i] or canto[i]: continue
        if riscos.get(i, ('',))[0] == 'apagar': continue
        if x0[i] < E0 or x0[i] + w[i] > E1 or y0[i] < T0 or y0[i] + h[i] > T1: continue
        if h[i] > 6 * xh and w[i] < 0.5 * xh: continue  # risco vertical (sombra da lombada)
        fino = w[i] <= 0.35 * xh and h[i] >= 0.6 * xh
        if fino and h[i] > 2.4 * xh: continue  # lasca: nenhuma letra é tão alta e fina
        if fino:
            # lasca solta: traço fino separado das letras dos dois lados (um "l" ou "i" sempre está numa palavra)
            dono = [b for b in bandas if b['n'] >= 3 and b['xt'] - 0.5 * xh <= cen[i][1] <= b['bl'] + 0.5 * xh
                    and b['x0'] - 0.6 * xh <= cen[i][0] <= b['x1'] + 0.6 * xh]
            if dono:
                b = dono[0]
                if 'seg' not in b:
                    b['seg'] = {}
                    for s in _segmentos([j for j in zona[id(b)] if a[j] >= 4], x0, w, xh):
                        tem = any(w[j] > 0.35 * xh for j in s)
                        for j in s: b['seg'][j] = tem
                if not b['seg'].get(i, True): continue  # só traços finos juntos, nenhuma letra cheia
                blf = _local(b, cen[i][0])
                if y0[i] + h[i] < blf - 0.6 * xh or y0[i] > blf + 0.1 * xh: continue  # traço solto acima da letra ou abaixo da linha
        pequeno = a[i] < 0.3 * xh * xh and w[i] < 0.6 * xh and h[i] < 0.6 * xh
        if pequeno or (h[i] < 0.4 * xh and w[i] < 2.5 * xh):
            cx, cy = cen[i]
            dono = [b for b in bandas if b['xt'] - 1.0 * xh <= cy <= b['bl'] + 0.6 * xh and b['x0'] - 0.6 * xh <= cx <= b['x1'] + 0.6 * xh]
            if not dono: continue  # entre linhas, ou fora de qualquer linha
            b = min(dono, key=lambda b: abs(cy - (b['bl'] + b['xt']) / 2))
            bl = _local(b, cx); xt = _local(b, cx, 'pt')
            if cy < xt:
                # marcas acima da altura de x: acento, pingo, til, aspas — ou sujeira
                baixo = [j for j in b['ids'] if x0[j] - 1 <= cx <= x0[j] + w[j] + 1 and y0[j] > y0[i]]
                if h[i] <= 0.3 * xh and w[i] >= 1.6 * h[i] and w[i] <= 1.2 * xh:
                    # traço horizontal: o espanhol não tem mácron; só o til do ñ, sobre uma letra em forma de n
                    if not any(0.8 * xh <= h[j] <= 1.25 * xh and _forma_n(lab, st, j) for j in baixo): continue
                if pequeno and not baixo:
                    # sem letra embaixo: aspas ficam coladas à palavra; marca solta no espaço é sujeira
                    viz = [j for j in b['ids'] if j != i]
                    esq = min([x0[i] - (x0[j] + w[j]) for j in viz if x0[j] + w[j] <= x0[i] + 1] or [99])
                    dirr = min([x0[j] - (x0[i] + w[i]) for j in viz if x0[j] >= x0[i] + w[i] - 1] or [99])
                    if esq > 0.3 * xh and dirr > 0.3 * xh: continue
            if pequeno and w[i] < 1.6 * h[i]:
                viz = [j for j in zona[id(b)] if j != i]
                esq = min([x0[i] - (x0[j] + w[j]) for j in viz if x0[j] + w[j] <= x0[i] + 1] or [99])
                dirr = min([x0[j] - (x0[i] + w[i]) for j in viz if x0[j] >= x0[i] + w[i] - 1] or [99])
                if bl - 0.68 * xh < cy < bl - 0.28 * xh:
                    # ponto no meio da altura: só é pontuação se houver outro ponto alinhado (dois-pontos, ponto e vírgula)
                    par = [j for j in viz if abs(cen[j][0] - cx) < 0.35 * xh and 0.25 * xh < abs(cen[j][1] - cy) < 1.0 * xh
                           and a[j] < 0.3 * xh * xh and (cen[j][1] > cy or a[j] < 2 * a[i])]
                    if not par: continue
                elif cy >= bl - 0.28 * xh and esq > 0.45 * xh and dirr > 0.45 * xh:
                    continue  # ponto solto no meio do espaço entre palavras (o ponto final encosta na letra)
            if y0[i] > bl + 0.05 * xh and w[i] > 1.3 * h[i] and h[i] < 0.35 * xh: continue  # traço sob a linha
        manter.append(i)
    # número de página: última linha, curta e afastada do texto
    folio = None
    if len(bandas) >= 2:
        u, p = bandas[-1], bandas[-2]
        if u['n'] <= 4 and u['x1'] - u['x0'] < 4 * xh and u['bl'] - p['bl'] > 2.5 * xh: folio = u['bl']
    # conferência: quanto das peças com tamanho de letra (dentro da página) ficou
    letras = set(np.where(letra & (h <= 1.6 * xh) & (w >= 0.4 * xh))[0])
    mset = set(manter)
    ficou = len(letras & mset) / max(1, len(letras))
    # letra removida encostada no texto que ficou (fim de linha cortado, por exemplo)
    cortadas = []
    for i in letras - mset:
        for b in bandas:
            if b['xt'] - 0.3 * xh <= cen[i][1] <= b['bl'] + 0.3 * xh and any(
                    j in mset and (0 <= x0[i] - (x0[j] + w[j]) <= 0.4 * xh or 0 <= x0[j] - (x0[i] + w[i]) <= 0.4 * xh) for j in b['ids']):
                cortadas.append(st[i][:4].tolist()); break
    return dict(lab=lab, manter=manter, bandas=bandas, caixa=(bx0, by0, bx1, by1), folio=folio, ficou=ficou, nletras=len(letras),
                aparar=[a for a in aparar if a[0] in mset], cortadas=cortadas)


def suavizar(clean, esc):
    """Amplia a imagem binária e devolve bordas com meio-tom (como uma impressão escaneada em cinza)."""
    big = cv2.resize(clean, None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    big = cv2.GaussianBlur(big, (0, 0), 0.45 * esc)
    return np.clip((big - 35) * 255.0 / (220 - 35), 0, 255).astype(np.uint8)


def main(pdf, pasta, esc=1.5):
    pasta = Path(pasta); (pasta / 'paginas').mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    folhas = [imagem(doc, i) for i in range(len(doc))]
    xh = xh_livro(folhas[len(folhas) // 3: len(folhas) // 3 + 4])
    print(f'{len(folhas)} folhas · altura de x ≈ {xh:.0f} px')
    metades = []  # (folha, lado, imagem com a lombada na borda interna)
    angs = []
    for n, g in enumerate(folhas):
        L = lombada(g)
        for lado, m in (('esq', g[:, :L]), ('dir', g[:, L:])):
            ang = inclinacao(m, xh)
            ang += inclinacao(endireitar(m, ang), xh)  # resíduo: gira o original uma vez só, pelo ângulo somado
            angs.append(ang)
            metades.append((n, lado, endireitar(m, ang)))
    print(f'inclinação corrigida: mediana {np.median(np.abs(angs)):.2f}°, máxima {np.max(np.abs(angs)):.2f}°')
    Wc = int(np.median([m[2].shape[1] for m in metades])); Hc = int(np.median([g.shape[0] for g in folhas]))
    an = [analisar(m[2], xh) for m in metades]
    for (n, lado, _), r in zip(metades, an):
        if r['nletras'] >= 30 and r['ficou'] < 0.95:
            print(f'  atenção: folha {n + 1} ({lado}) manteve só {r["ficou"]:.0%} das letras ({r["nletras"]}) - conferir')
        if r.get('cortadas'):
            print(f'  atenção: folha {n + 1} ({lado}): {len(r["cortadas"])} peça(s) com tamanho de letra removida(s) junto do texto {r["cortadas"][:4]}')
    # deslocamento vertical por folha (as duas metades vêm do mesmo escaneamento)
    fol = {}
    for (n, _, _), r in zip(metades, an):
        if r['folio'] is not None: fol.setdefault(n, []).append(r['folio'])
    alvo = float(np.median([v for vs in fol.values() for v in vs])) if fol else None
    dys = {n: alvo - float(np.mean(v)) for n, v in fol.items()} if fol else {}
    dy_padrao = float(np.median(list(dys.values()))) if dys else 0.0
    paginas = []
    for (n, lado, g), r in zip(metades, an):
        clean = np.full(g.shape, 255, np.uint8)
        if r['manter']: clean[np.isin(r['lab'], r['manter'])] = 0
        for i, ya, yb in r.get('aparar', []):  # letra atravessada pelo risco: tira só o que passa da altura de letra
            fora = r['lab'] == i
            fora[max(0, int(ya)):max(0, int(yb)) + 1] = False
            clean[fora] = 255
        dy = int(round(dys.get(n, dy_padrao)))
        canvas = np.full((Hc, Wc), 255, np.uint8)
        h, w = clean.shape
        # lombada na borda interna: página da esquerda encosta à direita, a da direita à esquerda
        sx0 = max(0, w - Wc) if lado == 'esq' else 0
        src = clean[:, sx0:sx0 + Wc]
        cx0 = Wc - src.shape[1] if lado == 'esq' else 0
        ys0, yd0 = max(0, -dy), max(0, dy)
        hh = min(h - ys0, Hc - yd0)
        if hh > 0: canvas[yd0:yd0 + hh, cx0:cx0 + src.shape[1]] = src[ys0:ys0 + hh]
        paginas.append(dict(img=canvas, vazia=not r['manter'], folha=n, lado=lado))
    while paginas and paginas[0]['vazia']: paginas.pop(0)  # verso da capa: o livro começa numa página da direita
    # anotações à mão (carimbos, números de biblioteca) não seguem regra: áreas a apagar, por página,
    # em <pasta>/apagar.json -> {"004": "tudo", "005": [[x0, y0, x1, y1], ...]} em pixels da página final
    fa = pasta / 'apagar.json'
    apagar = json.load(open(fa, encoding='utf-8')) if fa.exists() else {}
    for f in (pasta / 'paginas').glob('*.png'): f.unlink()
    for k, p in enumerate(paginas, 1):
        img = suavizar(p['img'], esc)
        for r in ([[0, 0, img.shape[1], img.shape[0]]] if apagar.get(f'{k:03d}') == 'tudo' else apagar.get(f'{k:03d}', [])):
            img[int(r[1]):int(r[3]), int(r[0]):int(r[2])] = 255
        cv2.imwrite(str(pasta / 'paginas' / f'{k:03d}_pagina.png'), img)
    print(f'{len(paginas)} páginas em {pasta / "paginas"} ({Wc * esc:.0f} × {Hc * esc:.0f} px)')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.5)
