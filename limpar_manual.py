"""Apaga a sujeira no espaço entre palavras nos casos julgados à mão.

Uso: py limpar_manual.py "<pasta do livro limpo>"
Lê   <pasta>/reconstrucao/limpeza_manual.json  {"limpar": [ids], "limpar_menos_pontuacao": [ids], "apagar_token": [ids]}
     (ids de triagem_julgada.json, a triagem sobre a qual se julgou; vale a posição na página, então pode
      rodar depois de refazer o OCR)
Apaga as marcas soltas entre os trechos de letras das palavras daquele lugar; em "limpar_menos_pontuacao",
fica o ponto ou a vírgula colado à palavra da esquerda; em "apagar_token", sai tudo o que está na caixa.
Registra em manchas_removidas.json.
"""
import sys, json
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2, numpy as np
import triagem as T

INK = 140


def marcas_internas(g, w, bl, xh, menos_pontuacao):
    X0, X1 = max(0, int(w['x'] - 2)), min(g.shape[1], int(w['x'] + w['w'] + 2))
    Y0, Y1 = max(0, int(bl - 1.9 * xh)), min(g.shape[0], int(bl + 0.8 * xh))
    sub = (g[Y0:Y1, X0:X1] < INK).astype(np.uint8)
    k, lab, st, cen = cv2.connectedComponentsWithStats(sub, 8)
    pecas = [j for j in range(1, k) if st[j][4] >= 4 and not (st[j][0] == 0 or st[j][0] + st[j][2] >= sub.shape[1])]
    pont = lambda j: st[j][4] < 0.18 * xh * xh and st[j][3] < 0.6 * xh
    corpos = sorted((j for j in pecas if not pont(j)), key=lambda j: st[j][0])
    if not corpos: return []
    trechos, fim = [], -1e9
    for j in corpos:
        if st[j][0] - fim >= 0.28 * xh: trechos.append([st[j][0], 0])
        fim = max(fim, st[j][0] + st[j][2]); trechos[-1][1] = fim
    out = []
    for j in pecas:
        if not pont(j): continue
        a, b = st[j][0], st[j][0] + st[j][2]
        if any(a < t1 and b > t0 for t0, t1 in trechos): continue      # sobre as letras: pingo, acento
        if a >= trechos[-1][1] or b <= trechos[0][0]: continue        # pontas do token: pontuação
        if menos_pontuacao:
            base = Y0 + st[j][1] + st[j][3] >= bl - 0.1 * xh
            if base and any(0 <= a - t1 <= 0.4 * xh for t0, t1 in trechos): continue  # . ou , colado à palavra
        x, y, ww, hh, _ = st[j]
        a0, b0 = max(0, x - 2), max(0, y - 2); a1, b1 = min(sub.shape[1], x + ww + 2), min(sub.shape[0], y + hh + 2)
        L_ = lab[b0:b1, a0:a1]
        m = cv2.dilate((L_ == j).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        m &= ~((L_ > 0) & (L_ != j))
        out.append((Y0 + b0, X0 + a0, m))
    return out


def palavras_no_lugar(pg, caixa):
    """Palavras do OCR atual que ocupam o lugar de uma caixa (de uma triagem antiga): o OCR pode ter separado
    em duas o que antes vinha junto, ou o contrário."""
    x, y, w, h = caixa
    out = []
    for i, p in enumerate(pg['words']):
        ix = min(x + w, p['x'] + p['w']) - max(x, p['x']); iy = min(y + h, p['y'] + p['h']) - max(y, p['y'])
        if ix > 0 and iy > 0 and ix * iy >= 0.3 * p['w'] * p['h']: out.append(i)
    return out


def main(pasta):
    pasta = Path(pasta); rd = pasta / 'reconstrucao'
    J = json.load(open(rd / 'limpeza_manual.json', encoding='utf-8-sig'))
    # as decisões citam ids de triagem_julgada.json (a triagem sobre a qual se julgou); vale a posição na página
    fref = rd / 'triagem_julgada.json'
    tri = {t['id']: t for t in json.load(open(fref if fref.exists() else rd / 'triagem.json', encoding='utf-8'))}
    ocr = json.load(open(pasta / 'ocr.json', encoding='utf-8'))
    hs = [c[4] - c[2] for pg in ocr.values() for c in pg['chars'] if c[0] in 'aceomnrsuvxz']
    xh = float(np.median(hs))
    acoes = ({i: 'limpar' for i in J.get('limpar', [])} | {i: 'pontuacao' for i in J.get('limpar_menos_pontuacao', [])}
             | {i: 'token' for i in J.get('apagar_token', [])})
    por_pag = {}
    for i, a in acoes.items(): por_pag.setdefault(tri[i]['pagina'], []).append((tri[i], a))
    removidas, n = [], 0
    for nome, casos in por_pag.items():
        pg = ocr[nome]; g = cv2.imread(str(pasta / 'paginas' / nome), cv2.IMREAD_GRAYSCALE)
        linhas = T.linhas_da_pagina(pg); bls = T.base_das_linhas(pg, linhas, xh)
        for t, a in casos:
            x, y, w, h = t['caixa']
            if a == 'token':  # o token inteiro é sujeira: tudo o que estiver dentro da caixa
                sub = g[y - 2:y + h + 2, x - 2:x + w + 2]; m = sub < 200
                if m.any():
                    sub[cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)] = 255; n += 1
                    removidas.append(dict(pagina=nome, palavra=t['texto'], caixa=[x - 2, y - 2, w + 4, h + 4], manual=True))
                continue
            ids = palavras_no_lugar(pg, t['caixa'])
            bl = next((bls[i] for i in ids if i in bls), None)
            if bl is None: continue
            ws = [pg['words'][i] for i in ids] or [dict(x=x, y=y, w=w, h=h)]
            x0 = min(p['x'] for p in ws); x1 = max(p['x'] + p['w'] for p in ws)
            junta = dict(x=x0, y=y, w=x1 - x0, h=h)  # as palavras do lugar, como um token só
            for yy, xx, m in marcas_internas(g, junta, bl, xh, a == 'pontuacao'):
                g[yy:yy + m.shape[0], xx:xx + m.shape[1]][m] = 255; n += 1
                removidas.append(dict(pagina=nome, palavra=t['texto'], caixa=[int(xx), int(yy), int(m.shape[1]), int(m.shape[0])], manual=True))
        cv2.imwrite(str(pasta / 'paginas' / nome), g)
    fr = rd / 'manchas_removidas.json'
    antes = json.load(open(fr, encoding='utf-8')) if fr.exists() else []
    json.dump(antes + removidas, open(fr, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{n} marcas apagadas em {len(acoes)} casos ({len(por_pag)} páginas).')


if __name__ == '__main__':
    main(sys.argv[1])
