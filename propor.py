"""Monta as propostas de reconstrução a partir da triagem.

Uso: py propor.py "<pasta do livro limpo>"
Lê   <pasta>/reconstrucao/triagem.json e, se existir, <pasta>/reconstrucao/julgamento.json
Grava <pasta>/reconstrucao/propostas.json (o que revisar.py mostra) e pendentes.json (casos sem decisão)

Regras automáticas:
  palavra certa com sujeira (imagem)  -> redesenhar a mesma palavra limpa (certeza alta)
  acento que é sujeira ("úna", "hábía") -> a forma sem o acento a mais, se for palavra (certeza alta)
  palavra certa só com confiança baixa -> nada a fazer: a imagem está boa (o OCR é que hesitou)
O resto (palavras grudadas por sujeira, letras trocadas, leituras sem sentido) depende de olhar a imagem
e o contexto: vai para julgamento.json -> {"id": {"proposta": ..., "certeza": ..., "nota": ...} ou "descartar"}.
"""
import sys, json, re, itertools, unicodedata
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from triagem import Lexico, nucleo, PONT, casar
import idioma
import collections

ACENTOS = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
           'â': 'a', 'ê': 'e', 'ô': 'o', 'à': 'a', 'Â': 'A', 'Ê': 'E', 'Ô': 'O', 'À': 'A'}   # til (ã, õ) não: muda a palavra


def sem_acento_a_mais(n, lex):
    """Tira acentos (um, dois...) até virar palavra; prefere a forma mais frequente com menos mudanças."""
    pos = [k for k, c in enumerate(n) if c in ACENTOS]
    for r in range(1, len(pos) + 1):
        cands = []
        for comb in itertools.combinations(pos, r):
            v = ''.join(ACENTOS[c] if k in comb else c for k, c in enumerate(n))
            if lex.ok(v): cands.append(v)
        if cands: return max(cands, key=lambda v: lex.f.get(v.lower(), 0))
    return None


def conferir_geometria(pasta, props, ocr, lex_f):
    """Rebaixa para 'média' as propostas de certeza alta em que o desenho não cabe bem: a caixa do OCR era
    larga demais (pegava a palavra vizinha) ou a palavra teve de ser espremida entre as vizinhas."""
    if not (pasta / 'glifos.pkl').exists(): return
    import cv2, uniio  # noqa: F401
    from glifos import Motor
    from casos import geometria, num_pagina
    from casos import grupos_colunas, INK
    motor = Motor(pasta); paginas = {}; n = n2 = 0; xh = motor.meta['xh']
    for p in props:
        if p['certeza'] == 'baixa': continue
        if p['pagina'] not in paginas: paginas[p['pagina']] = cv2.imread(str(pasta / 'paginas' / p['pagina']), cv2.IMREAD_GRAYSCALE)
        pg = paginas[p['pagina']]
        g = geometria(motor, ocr[p['pagina']], p, p['proposta'], num_pagina(p['pagina']), pg)
        larg, nat = p['caixa'][2], motor.largura(g['texto']); ww = g['x1'] - g['x0']
        # um espaço de palavra dentro da área e a proposta sem espaço: o OCR juntou duas palavras ("delas" = "de las")
        bl = g['baseline']
        faixa = pg[int(bl - xh):int(bl), int(g['x0']):int(g['x1'])] < INK
        gs = grupos_colunas(faixa)
        vao = max([b[0] - a[1] for a, b in zip(gs, gs[1:])] or [0])
        w_ = p['proposta'].lower()
        comum = lambda s: (len(s) >= 2 or s in 'ayoeu') and lex_f.get(s, 0) >= 5000
        partivel = any(comum(w_[:k]) and comum(w_[k:]) for k in range(1, len(w_)))
        if ' ' not in p['proposta'] and vao >= 0.2 * xh and partivel:
            p['certeza'] = 'baixa'; n2 += 1
            p['nota'] = (p.get('nota', '') + '; parece haver duas palavras (um espaço dentro): conferir').strip('; ')
        elif p['certeza'] == 'alta' and (larg > 1.3 * nat or (ww > 0 and abs(nat - ww) > 0.15 * ww)):
            p['certeza'] = 'média'; p['nota'] = (p.get('nota', '') + '; conferir o encaixe: a palavra não cabe bem no espaço').strip('; ')
            n += 1
    print(f'{n} propostas rebaixadas para certeza média pelo encaixe, {n2} para baixa (podem ser duas palavras)')


def main(pasta):
    pasta = Path(pasta); rd = pasta / 'reconstrucao'; idioma.definir(pasta)
    T = json.load(open(rd / 'triagem.json', encoding='utf-8'))
    fj = rd / 'julgamento.json'
    J0 = json.load(open(fj, encoding='utf-8-sig')) if fj.exists() else {}  # aceita arquivo salvo com BOM (Bloco de Notas)
    ignorar = set(J0.get('_paginas_ignoradas', []))
    # o julgamento foi feito sobre triagem_julgada.json; os ids mudam a cada rodada: liga pela posição
    fref = rd / 'triagem_julgada.json'
    J = {k: v for k, v in J0.items() if k.startswith('_')}
    if fref.exists():
        mapa = casar(json.load(open(fref, encoding='utf-8')), T)
        for k, v in J0.items():
            if not k.startswith('_') and int(k) in mapa: J[str(mapa[int(k)]['id'])] = v
        print(f'julgamento: {sum(not k.startswith("_") for k in J0)} decisões, {sum(not k.startswith("_") for k in J)} ligadas à triagem atual')
    else:
        J = J0
    ocr = json.load(open(pasta / 'ocr.json', encoding='utf-8'))
    cont = collections.Counter(nucleo(w['t']).lower() for pg in ocr.values() for w in pg['words'] if w['conf'] >= 90 and nucleo(w['t']).isalpha())
    lex = Lexico({w for w, n in cont.items() if n >= 2 and len(w) >= 2})
    props, pend, n_desc = [], [], collections.Counter()
    for t in T:
        k = str(t['id']); n = nucleo(t['texto'])
        if t['pagina'] in ignorar: n_desc['página de créditos/colofão'] += 1; continue
        j = J.get(k)
        if j == 'descartar': n_desc['julgado: imagem boa'] += 1; continue
        if isinstance(j, dict):
            props.append(dict(t, **j)); continue
        valida = n.isalpha() and lex.ok(n)
        if valida and 'imagem' in t['origem']:
            props.append(dict(t, proposta=n, certeza='alta', nota=(t.get('nota', '') + ' — redesenhar a palavra limpa').strip(' —')))
            continue
        if valida and t['origem'] == 'léxico':
            n_desc['palavra certa, só confiança baixa'] += 1; continue
        if n.isalpha() and any(c in ACENTOS for c in n):
            v = sem_acento_a_mais(n, lex)
            if v:
                # o acento tirado é o da última letra e a forma acentuada existe: pode ser verbo no pretérito ("aleteó")
                final = v[-1] != n[-1] and lex.f.get(n.lower(), 0) >= 3
                props.append(dict(t, proposta=v, certeza='baixa' if final else ('alta' if t['conf'] >= 50 else 'média'),
                                  nota=('acento na última letra: pode ser verbo no pretérito (aleteó, llamó); conferir o contexto'
                                        if final else 'sujeira lida como acento') + (f'; {t["nota"]}' if t.get('nota') else '')))
                continue
        pend.append(t)
    for i, p in enumerate(props, 1): p.setdefault('certeza', 'média')
    conferir_geometria(pasta, props, ocr, lex.f)
    json.dump(props, open(rd / 'propostas.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(pend, open(rd / 'pendentes.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{len(props)} propostas (' + ', '.join(f'{c} {v}' for c, v in collections.Counter(p["certeza"] for p in props).items()) + ')')
    print('descartados: ' + ', '.join(f'{c} {v}' for c, v in n_desc.items()))
    print(f'{len(pend)} pendentes de julgamento (em {rd / "pendentes.json"})')


if __name__ == '__main__':
    main(sys.argv[1])
