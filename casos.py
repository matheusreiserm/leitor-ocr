"""Geometria de cada caso (linha de base, limites, área a limpar) medida na própria imagem, e prévia ANTES/DEPOIS."""
import json
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
from pathlib import Path
import cv2, numpy as np
from glifos import Motor

INK = 140
PONT_FIM = '?!.,;:…"”»)]\'’*-'
PONT_INI = '¿¡"“«([\'‘—'
DESC = set('gjpqyQ,;()')


def _npont(s, fim):
    """Quantos sinais de pontuação (em 'grupos de tinta') há no fim/início do texto do OCR."""
    t = s if fim else s[::-1]
    n = 0
    for c in reversed(t):
        if c in (PONT_FIM if fim else PONT_INI): n += 3 if c == '…' else 1
        else: break
    return n


def baseline(ocr_pg, w, xh):
    """Linha de base: pé das palavras vizinhas na mesma altura visual que não têm descendentes."""
    cy = w['y'] + w['h'] / 2
    pes = []
    for v in ocr_pg['words']:
        t = v['t']
        if abs((v['y'] + v['h'] / 2) - cy) > 0.6 * xh or abs(v['x'] - w['x']) > 900: continue
        if any(c in DESC for c in t) or not any(c.isalpha() for c in t): continue
        pes.append(v['y'] + v['h'])
    if pes: return float(np.median(pes))
    return float(w['y'] + w['h'] - (0.45 * xh if any(c in DESC for c in w['t']) else 0))


def grupos_colunas(ink):
    cols = ink.any(axis=0)
    gs, ini = [], None
    for i, c in enumerate(cols):
        if c and ini is None: ini = i
        if not c and ini is not None: gs.append((ini, i)); ini = None
    if ini is not None: gs.append((ini, len(cols)))
    return gs


def geometria(motor, ocr_pg, cand, texto, pagina_num, page):
    w = ocr_pg['words'][cand['i']]
    X, Y, W, H = cand['caixa']
    xh = motor.meta['xh']
    texto = texto.strip('¿?¡!.,;:…"“”«»()[]—–\'’‘*')
    bl = baseline(ocr_pg, w, xh)
    top, bot = int(bl - 1.6 * xh), int(bl + 0.65 * xh)
    a, b = max(0, X - 3), min(page.shape[1], X + W + 3)
    ink = page[max(0, top):bot, a:b] < INK
    gs = grupos_colunas(ink)
    pf, pi = _npont(cand['texto'], True), _npont(cand['texto'], False)
    tem_pf = tem_pi = False
    corte_fim = 0.0
    if motor.cfg.get('geometria', 1) >= 2 and gs:
        # pontuação reconhecida pela forma (a caixa do OCR às vezes inclui aspas que o texto não traz):
        # grupo de colunas todo acima da altura de x = aspa/apóstrofo; todo embaixo e estreito = ponto/vírgula
        def pont(g):
            rows = np.where(ink[:, g[0]:g[1]].any(axis=1))[0] + max(0, top)
            if not len(rows): return True
            alto = rows.max() < bl - 0.9 * xh
            # estreito e sem chegar ao topo da altura de x (toda minúscula chega): . , ; : -
            baixo = rows.min() > bl - 0.75 * xh and g[1] - g[0] <= 0.45 * xh
            # pouca tinta (menos que um "i"): ponto com uma marca solta em cima, aspas, sujeira
            pouca = int(ink[:, g[0]:g[1]].sum()) <= 0.17 * xh * xh and g[1] - g[0] <= 0.9 * xh
            return alto or baixo or pouca
        while len(gs) > 1 and pont(gs[-1]): gs = gs[:-1]; tem_pf = True
        while len(gs) > 1 and pont(gs[0]): gs = gs[1:]; tem_pi = True
        pf = 0 if tem_pf else pf; pi = 0 if tem_pi else pi
        # caixa larga demais (o OCR juntou palavras vizinhas): para no espaço depois da largura natural da palavra
        nat0 = motor.largura(texto)
        if gs and gs[-1][1] - gs[0][0] > 1.12 * nat0:
            fim = gs[0][1]
            for g0, g1 in gs[1:]:
                if g0 - fim >= 0.3 * xh and fim - gs[0][0] >= 0.8 * nat0: break
                fim = g1
            gs = [g for g in gs if g[1] <= fim]
    if len(gs) > pf + pi:
        if pf:
            # pontuação é estreita; se o último grupo é largo, a pontuação está colada na letra: corta só a largura dela
            larg = {'?': 0.62, '!': 0.38, ':': 0.36, ';': 0.36, '»': 0.6, '”': 0.55, '"': 0.55, ')': 0.4}.get(cand['texto'].rstrip()[-1:], 0.33)
            if pf == 1 and gs[-1][1] - gs[-1][0] > (larg + 0.12) * xh: corte_fim = larg * xh
            else: gs = gs[:-pf]
            tem_pf = True
        if pi: gs = gs[pi:]; tem_pi = True
    x0, x1 = (a + gs[0][0], a + gs[-1][1] - corte_fim) if gs else (X, X + W)
    nat = motor.largura(texto)
    ww = x1 - x0
    if nat > ww * 1.08:  # faltam letras: cresce para o lado da lombada (página ímpar: esquerda; par: direita)
        ancora = 'dir' if pagina_num % 2 == 1 else 'esq'
    elif nat < ww * 0.88:  # área maior que a palavra (mancha/outra palavra junto): largura natural, a partir do início
        ancora = 'esq'
    else:
        ancora = 'ajustar'
    if motor.cfg.get('geometria', 1) >= 2 and 0.8 * ww <= nat <= 1.25 * ww:
        # sem letras cortadas na margem (neste livro), a palavra redesenhada cabe no espaço que já ocupa:
        # não avança sobre a pontuação nem sobre as palavras do lado
        ancora = 'ajustar'
    if ancora == 'dir': nx0, nx1 = x1 - nat, x1
    elif ancora == 'esq': nx0, nx1 = x0, x0 + nat
    else: nx0, nx1 = x0, x1
    m0 = 0 if tem_pi else 2
    m1 = 0 if tem_pf else 2
    r0, r1 = int(min(x0, nx0) - m0), int(max(x1, nx1) + m1)
    rt, rb = int(bl - 1.55 * xh), int(bl + 0.62 * xh)
    if motor.cfg.get('geometria', 1) >= 2:
        # nunca encostar na tinta das palavras do lado (nem na pontuação que ficou de fora): se não couber, ajusta
        faixa = page[max(0, rt):rb] < INK
        cols = np.where(faixa.sum(axis=0) >= 2)[0]
        esq_ = cols[(cols < x0 - 1) & (cols >= x0 - 3 * xh)]; dir_ = cols[(cols > x1 + 1) & (cols <= x1 + 3 * xh)]
        lim0 = min(esq_.max() + 0.25 * xh, x0 - 1) if len(esq_) else -1e9  # nunca dentro da própria palavra
        lim1 = max(dir_.min() - 0.25 * xh, x1 + 1) if len(dir_) else 1e9
        if nx0 < lim0 or nx1 > lim1:
            nx0, nx1 = max(nx0, lim0), min(nx1, lim1); ancora = 'ajustar'; x0, x1 = nx0, nx1
        r0, r1 = int(max(r0, lim0)), int(min(r1, lim1))
        if ancora == 'esq': r1 = int(min(r1, nx1 + 0.2 * xh))  # caixa larga demais: não apaga além do fim natural da palavra
    # máscara de limpeza: o retângulo + manchas/restos conectados que encostam nele (sem invadir outras palavras)
    mask = np.zeros(page.shape, bool)
    mask[max(0, rt):rb, max(0, r0):r1] = True
    E0, E1, ET, EB = r0 - 0.35 * xh, r1 + 0.35 * xh, rt - 1.0 * xh, rb + 0.4 * xh
    ya, yb, xa, xb = max(0, int(ET)), min(page.shape[0], int(EB) + 1), max(0, int(E0)), min(page.shape[1], int(E1) + 1)
    sub = (page[ya:yb, xa:xb] < 200).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    for i in range(1, k):
        cx, cy, cw, ch, _a = st[i]
        gx0, gy0, gx1, gy1 = xa + cx, ya + cy, xa + cx + cw, ya + cy + ch
        toca = gx1 > r0 and gx0 < r1 and gy1 > rt and gy0 < rb
        dentro = gx0 >= E0 and gx1 <= E1 and gy0 >= ET and gy1 <= EB
        if motor.cfg.get('geometria', 1) >= 2 and gy1 < bl - 0.6 * xh and (gx0 >= x1 - 1 or gx1 <= x0 + 1):
            continue  # aspa/apóstrofo ao lado da palavra: não é da palavra
        if toca and dentro: mask[ya:yb, xa:xb][lab == i] = True
    if motor.cfg.get('geometria', 1) >= 2:
        # só dentro da faixa horizontal da palavra: um ponto colado à última letra (mesma peça) fica
        mask[:, :max(0, r0)] = False; mask[:, max(0, r1):] = False
    return dict(baseline=bl, x0=x0, x1=x1, ancora=ancora, apagar=mask, texto=texto, caixa=(r0, rt, r1, rb))


def num_pagina(nome):
    return int(nome.split('_')[0])


def previa(pasta, motor, ocr, cand, texto, zoom=1.0):
    """Imagem com a linha ANTES (em cima) e DEPOIS (embaixo)."""
    pasta = Path(pasta)
    page = cv2.imread(str(pasta / 'paginas' / cand['pagina']), cv2.IMREAD_GRAYSCALE)
    pg = ocr[cand['pagina']]
    geo = geometria(motor, pg, cand, texto, num_pagina(cand['pagina']), page)
    novo = motor.renderizar(page, geo['texto'], geo['baseline'], geo['x0'], geo['x1'], geo['ancora'], geo['apagar'])
    xh = motor.meta['xh']; bl = int(geo['baseline'])
    y0, y1 = max(0, int(bl - 3.6 * xh)), min(page.shape[0], int(bl + 2.4 * xh))
    a0, a1 = max(0, int(geo['caixa'][0] - 9 * xh)), min(page.shape[1], int(geo['caixa'][2] + 9 * xh))
    A, D = page[y0:y1, a0:a1], novo[y0:y1, a0:a1]
    def rot(img, txt):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        barra = np.full((34, img.shape[1], 3), 235, np.uint8); cv2.putText(barra, txt, (8, 24), 0, 0.75, (40, 40, 40), 2)
        return np.vstack([barra, img])
    out = np.vstack([rot(A, 'ANTES'), np.full((6, A.shape[1], 3), 120, np.uint8), rot(D, 'DEPOIS')])
    if zoom != 1.0: out = cv2.resize(out, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
    return out, novo, geo
