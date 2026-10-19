"""Triagem: palavras que precisam de revisão antes da reconstrução.

Uso: py triagem.py "<pasta do livro limpo>"
Lê   <pasta>/ocr.json e as páginas
Grava <pasta>/reconstrucao/triagem.json, com dois tipos de caso:
  léxico  - a leitura do OCR não é palavra do espanhol (nem do próprio livro), ou a confiança é baixa
  imagem  - a palavra foi lida, mas há sujeira grudada nela: tinta acima de letra que não leva acento
            nem tem haste, ou abaixo de letra sem perna (a reconstrução redesenha a palavra limpa)
"""
import sys, json, gzip, re, unicodedata, collections
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2, numpy as np
from spellchecker import SpellChecker

AQUI = Path(__file__).parent
PONT = '¿?¡!.,;:…"“”«»()[]—–-\'’‘*'
INK = 140
BAIXA = set('acemnorsuvwxz')          # sem haste, sem acento, sem perna
SEM_PERNA = set('abcdefhiklmnorstuvwxzáéíóúñABCDEFGHIJKLMNOPRSTUVWXZÁÉÍÓÚÑ')


def tira_acento(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


class Lexico:
    def __init__(self, livro):
        self.f = json.load(gzip.open(AQUI / 'lexico' / 'es.json.gz', 'rt', encoding='utf-8'))
        self.semac = collections.defaultdict(int)
        for w, n in self.f.items(): self.semac[tira_acento(w)] = max(self.semac[tira_acento(w)], n)
        self.livro = livro  # palavras que o próprio livro confirma (nomes, termos raros)
        self.sp = SpellChecker(language=None, distance=1); self.sp.word_frequency.load_dictionary(str(AQUI / 'lexico' / 'es.json.gz'))

    def ok(self, w):
        lw = w.lower()
        if lw in self.livro: return True
        n = self.f.get(lw, 0)
        if n < 3: return False
        if lw != tira_acento(lw) and n < 300 and self.f.get(tira_acento(lw), 0) > 100 * n:
            return False  # acento raro sobre palavra comum ("esás", "ló"): quase sempre é sujeira lida como acento
        return True

    def sugestoes(self, w):
        lw = w.lower()
        c = {x for x in (self.sp.candidates(lw) or set()) if x in self.f}
        if not c and len(lw) <= 8:  # distância 2 só para palavras curtas (nas longas, o número de variantes explode)
            self.sp.distance = 2
            c = {x for x in (self.sp.candidates(lw) or set()) if x in self.f}
            self.sp.distance = 1
        c = sorted((x for x in c if x != lw), key=lambda x: -self.f.get(x, 0))[:4]
        return [x.capitalize() if w[:1].isupper() else x for x in c]


def nucleo(t):
    return t.strip(PONT)


def linhas_da_pagina(pg):
    L = collections.OrderedDict()
    for i, w in enumerate(pg['words']): L.setdefault(tuple(w['line']), []).append(i)
    return list(L.values())


MARCAS = {**{c: 1 for c in 'iíjáéóúñüÁÉÍÓÚÑÜ¡¿’‘\''}, **{c: 2 for c in '“”"'}}  # peças soltas acima da altura de x


COM_MARCA = {**{c: 1 for c in 'iíjáéóúñÁÉÍÓÚÑ'}, 'ü': 2, 'Ü': 2}  # marcas que cada letra leva


def manchas(gray, pg, idxs, xh, linhas_bl):
    """Sujeira grudada na palavra, contando as peças de tinta (as caixas de letra do Tesseract são imprecisas).
    Devolve {palavra: (achados, apagaveis, ambiguo)}:
      acima da altura de x, marcas soltas a mais do que a palavra lida pede (pingos, acentos, til, aspas);
      abaixo da linha de base, qualquer peça solta (as pernas de g, p, y são ligadas à letra).
    apagaveis: peças que com certeza são sujeira -> [(y0, x0, máscara)]; ambiguo: sobram marcas sem dono certo."""
    H, W = gray.shape
    out = {}
    for i in idxs:
        w = pg['words'][i]
        bl = linhas_bl.get(i)
        if bl is None: continue
        xt = bl - xh
        X0, X1 = max(0, int(w['x'] - 0.3 * xh)), min(W, int(w['x'] + w['w'] + 0.3 * xh))
        Y0, Y1 = max(0, int(bl - 1.9 * xh)), min(H, int(bl + 0.8 * xh))
        sub = (gray[Y0:Y1, X0:X1] < INK).astype(np.uint8)
        k, lab, st, cen = cv2.connectedComponentsWithStats(sub, 8)
        cima, baixo, corpos = [], [], []
        for j in range(1, k):
            x, y, ww, hh, a = st[j]
            gx, gy0, gy1 = X0 + cen[j][0], Y0 + y, Y0 + y + hh
            if a < 6 or not (w['x'] - 0.2 * xh <= gx <= w['x'] + w['w'] + 0.2 * xh): continue
            if x == 0 or x + ww >= sub.shape[1]: continue  # peça da palavra vizinha
            corpo = hh >= 0.6 * xh or gy1 >= bl - 0.5 * xh
            if corpo: corpos.append(j)
            elif Y0 + cen[j][1] < xt: cima.append(j)
            elif gy0 > bl + 0.3 * xh: baixo.append(j)  # bem abaixo: em linha curta a base estimada erra uns pixels
        # o que vem logo depois da última letra é pontuação (ponto, vírgula), mesmo que pareça baixo
        if corpos:
            fim = max(st[c][0] + st[c][2] for c in corpos)
            baixo = [j for j in baixo if st[j][0] + st[j][2] / 2 < fim]
        espera = sum(MARCAS.get(c, 0) for c in w['t'])
        if len(cima) <= espera and not baixo: continue
        sujas = list(baixo)
        ambiguo = False
        if len(cima) > espera:
            letras = [c for c in w['t'] if c.isalpha()]
            corpos.sort(key=lambda j: st[j][0])
            aspas = sum(MARCAS.get(c, 0) for c in w['t'] if not c.isalpha())
            # só é sujeira certa a marca que está sobre uma letra; ao lado da palavra podem ser aspas que o OCR não leu
            letras_ = [c for c in corpos if st[c][4] >= 0.18 * xh * xh]  # a vírgula do ";" não é letra
            sobre = lambda m: any(st[c][0] - 1 <= cen[m][0] <= st[c][0] + st[c][2] + 1 for c in letras_)  # o centro da marca sobre a letra
            if espera == 0:
                sujas += [m for m in cima if sobre(m)]
                ambiguo = any(not sobre(m) for m in cima)
            elif len(corpos) == len(letras) and aspas == 0:
                # cada peça-corpo é uma letra: a marca pertence à letra que está embaixo dela
                vagas = [COM_MARCA.get(c, 0) for c in letras]
                for m in sorted(cima, key=lambda j: st[j][0]):
                    if not sobre(m): continue
                    mx = cen[m][0]
                    dono = min(range(len(corpos)), key=lambda q: abs(st[corpos[q]][0] + st[corpos[q]][2] / 2 - mx))
                    if vagas[dono] > 0: vagas[dono] -= 1
                    else: sujas.append(m)
            else:
                ambiguo = True
        achados = []
        if len(cima) > espera: achados.append(f'{len(cima) - espera} marca(s) a mais sobre a palavra')
        if baixo: achados.append(f'{len(baixo)} traço(s) sob a palavra')
        apag = []
        for j in sujas:
            x, y, ww, hh, _ = st[j]
            # 2 px a mais em volta, para levar junto a borda suavizada (sem tocar na tinta das outras peças)
            a0, b0 = max(0, x - 2), max(0, y - 2); a1, b1 = min(sub.shape[1], x + ww + 2), min(sub.shape[0], y + hh + 2)
            L_ = lab[b0:b1, a0:a1]
            m = cv2.dilate((L_ == j).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
            m &= ~((L_ > 0) & (L_ != j))
            apag.append((Y0 + b0, X0 + a0, m))
        out[i] = (achados, apag, ambiguo)
    return out


IMPOSSIVEIS = set('<>*|~^_=+#@\\/')


def ruido_nos_espacos(gray, pg, idxs, xh, linhas_bl):
    """Marcas soltas num espaço entre palavras (o OCR então junta as duas: "no-había", "pero'el", "y.en").
    Fica a pontuação colada às letras ("pausa," "¡ya"); sai o que está solto num espaço de verdade.
    Devolve {palavra: [(y0, x0, máscara)]}."""
    H, W = gray.shape
    out = {}
    for i in idxs:
        w = pg['words'][i]; t = w['t']
        bl = linhas_bl.get(i)
        if bl is None: continue
        so_simbolo = all(c in IMPOSSIVEIS or c in '-—–.,;:' for c in t) and any(c in IMPOSSIVEIS for c in t)
        if not so_simbolo and (t.isalpha() or not any(c.isalpha() for c in t)): continue
        X0, X1 = max(0, int(w['x'] - 2)), min(W, int(w['x'] + w['w'] + 2))
        Y0, Y1 = max(0, int(bl - 1.9 * xh)), min(H, int(bl + 0.8 * xh))
        sub = (gray[Y0:Y1, X0:X1] < INK).astype(np.uint8)
        k, lab, st, cen = cv2.connectedComponentsWithStats(sub, 8)
        pecas = [j for j in range(1, k) if st[j][4] >= 4 and not (st[j][0] == 0 or st[j][0] + st[j][2] >= sub.shape[1])]
        if so_simbolo:
            sujas = [j for j in pecas if st[j][3] < 1.6 * xh]
        else:
            pont = lambda j: st[j][4] < 0.18 * xh * xh and st[j][3] < 0.6 * xh  # ponto, vírgula, traço, apóstrofo (haste de letra é mais alta)
            corpos = sorted((j for j in pecas if not pont(j)), key=lambda j: st[j][0])
            if not corpos: continue
            # trechos de letras separados por espaço de verdade (>= 0,28 x; entre letras o espaço é de 1 a 3 px)
            trechos, fim = [], -1e9
            for j in corpos:
                if st[j][0] - fim >= 0.28 * xh: trechos.append([st[j][0], 0])
                fim = max(fim, st[j][0] + st[j][2]); trechos[-1][1] = fim
            soltas = [j for j in pecas if pont(j) and not any(st[j][0] < t1 and st[j][0] + st[j][2] > t0 for t0, t1 in trechos)]
            cy = {j: Y0 + cen[j][1] for j in pecas}
            par = lambda j: any(q != j and abs(cy[q] - cy[j]) < 0.3 * xh and 0 < abs(cen[q][0] - cen[j][0]) < 0.5 * xh for q in soltas)
            # dois-pontos e ponto e vírgula: o par pode ser qualquer peça pequena (a vírgula do ";" às vezes não conta como solta)
            empilhada = lambda j: any(q != j and st[q][4] < 0.3 * xh * xh and abs(cen[q][0] - cen[j][0]) < 0.3 * xh
                                      and 0.3 * xh < abs(cy[q] - cy[j]) < 1.0 * xh for q in pecas)
            legit = set()
            # no começo e no fim do token fica tudo o que estiver encadeado à palavra (aspas, ponto, vírgula, ¿ ¡ ...):
            # o OCR só junta duas palavras num token por causa de sujeira ENTRE elas
            p0, p1 = trechos[0][0], trechos[-1][1]
            peq = [j for j in pecas if pont(j)]
            for _ in range(5):
                # a corrente inclui qualquer peça pequena (as aspas que encostam na última letra também)
                for j in peq:
                    a, b = st[j][0], st[j][0] + st[j][2]
                    if a < p1 + 0.6 * xh and b > p1: p1 = b
                    if b > p0 - 0.6 * xh and a < p0: p0 = a  # a aspa de abertura fica a uns 10 px da letra
                for j in soltas:
                    a, b = st[j][0], st[j][0] + st[j][2]
                    if j not in legit and (a >= trechos[-1][1] and b <= p1 or b <= trechos[0][0] and a >= p0): legit.add(j)
            for _ in range(4):  # voltas seguintes: pontuação colada a outra pontuação legítima ("," + "”", reticências)
                for j in soltas:
                    if j in legit: continue
                    a, b = st[j][0], st[j][0] + st[j][2]
                    if a >= trechos[-1][1] or b <= trechos[0][0]: continue  # ponta do token, longe da palavra: sujeira
                    alta = cy[j] < bl - xh + 0.1 * xh
                    base = cy[j] >= bl - 0.35 * xh or Y0 + st[j][1] + st[j][3] >= bl - 0.1 * xh  # a vírgula tem o centro alto, mas o pé na linha
                    esq = [(t0, t1) for t0, t1 in trechos] + [(st[q][0], st[q][0] + st[q][2]) for q in legit]
                    depois_de = any(0 <= a - t1 <= 0.35 * xh for t0, t1 in esq)       # logo depois de uma palavra
                    antes_de = any(0 <= t0 - b <= 0.35 * xh for t0, t1 in trechos)    # logo antes de uma palavra
                    segue = any(t0 > b for t0, t1 in trechos)                           # há outra palavra à direita
                    composta = (not alta and not base and any(0 <= a - t1 <= 0.15 * xh for t0, t1 in trechos)
                                and any(0 <= t0 - b <= 0.15 * xh for t0, t1 in trechos))  # hífen de "best-sellers": justo dos dois lados
                    justo = any(0 <= a - t1 <= 0.2 * xh for t0, t1 in esq)  # o ponto final encosta na letra
                    # pontuação de verdade é seguida (ou, a de abertura, precedida) de um espaço normal;
                    # a sujeira fica no meio de um espaço estreito, com folga pequena dos dois lados
                    folga_dir = min([t0 - b for t0, t1 in trechos if t0 >= b] or [99])
                    folga_esq = min([a - t1 for t0, t1 in trechos if t1 <= a] or [99])
                    espaco_depois, espaco_antes = folga_dir >= 0.3 * xh, folga_esq >= 0.3 * xh
                    virgula = base and st[j][3] >= 0.35 * xh and Y0 + st[j][1] + st[j][3] > bl + 0.15 * xh  # cabeça e cauda abaixo da linha
                    perto = any(0 <= a - t1 <= 0.5 * xh for t0, t1 in esq)
                    aspas = alta  # ‘ ’ simples (este livro) ou “ ”; no meio de um espaço estreito a folga decide
                    travessao = not alta and not base and st[j][2] >= 0.9 * xh  # — de diálogo ou de inciso
                    if travessao or composta or (justo and virgula) or (perto and empilhada(j)) or depois_de and espaco_depois and (
                            base or empilhada(j) or aspas or (not alta and not base and not segue)):
                        legit.add(j)  # . , ; : ” e o hífen de fim de linha
                    elif antes_de and espaco_antes and (aspas or st[j][2] >= 1.0 * xh):
                        legit.add(j)  # “ e o travessão de diálogo
            sujas = [j for j in soltas if j not in legit]
        apag = []
        for j in sujas:
            x, y, ww, hh, _ = st[j]
            a0, b0 = max(0, x - 2), max(0, y - 2); a1, b1 = min(sub.shape[1], x + ww + 2), min(sub.shape[0], y + hh + 2)
            L_ = lab[b0:b1, a0:a1]
            m = cv2.dilate((L_ == j).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
            m &= ~((L_ > 0) & (L_ != j))
            apag.append((Y0 + b0, X0 + a0, m))
        if apag: out[i] = apag
    return out


def base_das_linhas(pg, linhas, xh):
    """Linha de base de cada palavra: mediana do pé das caixas das letras sem haste da mesma linha."""
    chars = pg['chars']; out = {}
    if not chars: return out
    cx = np.array([(c[1] + c[3]) / 2 for c in chars]); cy = np.array([(c[2] + c[4]) / 2 for c in chars])
    for L in linhas:
        pes, alt = [], []
        for i in L:
            w = pg['words'][i]
            sel = np.where((cx >= w['x'] - 2) & (cx <= w['x'] + w['w'] + 2) & (cy >= w['y'] - 2) & (cy <= w['y'] + w['h'] + 2))[0]
            pes += [chars[k][4] for k in sel if chars[k][0] in BAIXA]
            alt += [chars[k][4] - chars[k][2] for k in sel if chars[k][0] in 'aceomnrsuvxz']
        # letra miúda (créditos, colofão): as medidas do texto corrido não valem ali
        if pes and (not alt or np.median(alt) >= 0.8 * xh):
            for i in L: out[i] = float(np.median(pes))
    return out


def casar(ref, atual, tol=18):
    """Liga casos de uma triagem antiga (ref) aos da atual pela página e pela posição: os ids mudam a cada rodada.
    Devolve {id_ref: caso_atual}."""
    por_pag = collections.defaultdict(list)
    for t in atual: por_pag[t['pagina']].append(t)
    cen = lambda t: (t['caixa'][0] + t['caixa'][2] / 2, t['caixa'][1] + t['caixa'][3] / 2)
    out = {}
    for r in ref:
        rx, ry = cen(r); melhor = None
        for t in por_pag[r['pagina']]:
            tx, ty = cen(t); d = abs(tx - rx) + abs(ty - ry) - (5 if t['texto'] == r['texto'] else 0)
            if abs(tx - rx) <= tol * 2 and abs(ty - ry) <= tol and (melhor is None or d < melhor[0]): melhor = (d, t)
        if melhor: out[r['id']] = melhor[1]
    return out


def main(pasta, limpar=False):
    pasta = Path(pasta); rd = pasta / 'reconstrucao'; rd.mkdir(exist_ok=True)
    ocr = json.load(open(pasta / 'ocr.json', encoding='utf-8'))
    removidas = []
    if limpar:
        import shutil
        bak = pasta / 'paginas_antes_manchas'
        if not bak.exists(): shutil.copytree(pasta / 'paginas', bak)
    # vocabulário confirmado pelo próprio livro: lido ao menos 2 vezes com confiança alta
    cont = collections.Counter(nucleo(w['t']).lower() for pg in ocr.values() for w in pg['words'] if w['conf'] >= 90 and nucleo(w['t']).isalpha())
    lex = Lexico({w for w, n in cont.items() if n >= 2 and len(w) >= 2})
    # altura de x do livro (em pixels da página final), pelas letras sem haste
    hs = [c[4] - c[2] for pg in ocr.values() for c in pg['chars'] if c[0] in 'aceomnrsuvxz']
    xh = float(np.median(hs)) if hs else 22.0
    print(f'altura de x ≈ {xh:.1f} px')
    casos, nid = [], 0
    for nome, pg in ocr.items():
        gray = cv2.imread(str(pasta / 'paginas' / nome), cv2.IMREAD_GRAYSCALE)
        linhas = linhas_da_pagina(pg)
        txt = [' '.join(pg['words'][i]['t'] for i in L) for L in linhas]
        texto_ok = {}
        # hifenização no fim da linha: "contradicién-" + "dose" valem juntas
        for k, L in enumerate(linhas):
            u = pg['words'][L[-1]]['t']
            if u.endswith('-') and k + 1 < len(linhas):
                p = pg['words'][linhas[k + 1][0]]['t']
                junta = nucleo(u) + nucleo(p)
                if junta.isalpha() and lex.ok(junta): texto_ok[L[-1]] = texto_ok[linhas[k + 1][0]] = junta
        todas = [i for L in linhas for i in L]
        bls = base_das_linhas(pg, linhas, xh)
        mch = manchas(gray, pg, todas, xh, bls)
        if limpar:
            n0 = len(removidas)
            esp = ruido_nos_espacos(gray, pg, todas, xh, bls)
            # (os espaços entre palavras que o OCR separou certo ficam como estão: nesta fonte o ponto final fica
            #  a 5-7 px da letra, a mesma distância de uma manchinha solta, e não há como distinguir sem arriscar o texto)
            for i in set(mch) | set(esp):
                apag = (mch[i][1] if i in mch else []) + esp.get(i, [])
                for y, x, m in apag:
                    gray[y:y + m.shape[0], x:x + m.shape[1]][m] = 255
                    removidas.append(dict(pagina=nome, palavra=pg['words'][i]['t'], caixa=[int(x), int(y), int(m.shape[1]), int(m.shape[0])]))
            if len(removidas) > n0: cv2.imwrite(str(pasta / 'paginas' / nome), gray)
            continue
        # o que a limpeza automática não resolve (marca sem dono certo) vira caso de revisão
        mch = {i: v[0] for i, v in mch.items() if v[2]}
        for k, L in enumerate(linhas):
            for i in L:
                w = pg['words'][i]; t = w['t']; n = nucleo(t)
                if not n: continue
                origem, nota = [], None
                so_num = re.fullmatch(r'[\d.,/\-–()]+', t) is not None
                if i not in texto_ok and not so_num:
                    estranho = not n.replace('-', '').isalpha() or not lex.ok(n.replace('-', ''))
                    if estranho or w['conf'] < 45: origem.append('léxico')
                if i in mch and not so_num:
                    origem.append('imagem'); nota = '; '.join(mch[i])
                if not origem: continue
                nid += 1
                c = dict(id=nid, pagina=nome, i=i, texto=t, conf=round(w['conf']), caixa=[w['x'], w['y'], w['w'], w['h']],
                         linha=txt[k], antes=txt[k - 1] if k else '', depois=txt[k + 1] if k + 1 < len(txt) else '',
                         origem='+'.join(origem))
                if 'léxico' in origem: c['sugestoes'] = lex.sugestoes(n) if n.isalpha() else []
                if nota: c['nota'] = nota
                casos.append(c)
    if limpar:
        fr = rd / 'manchas_removidas.json'  # acumula as passadas
        antes = json.load(open(fr, encoding='utf-8')) if fr.exists() else []
        json.dump(antes + removidas, open(fr, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'{len(removidas)} manchas apagadas em {len({r["pagina"] for r in removidas})} páginas '
              f'(registro em {rd / "manchas_removidas.json"}; páginas anteriores em paginas_antes_manchas). Refaça o OCR.')
        return
    json.dump(casos, open(rd / 'triagem.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    cnt = collections.Counter(c['origem'] for c in casos)
    print(f'{len(casos)} casos: ' + ', '.join(f'{k} {v}' for k, v in cnt.most_common()) + f'. Gravado em {rd / "triagem.json"}')


if __name__ == '__main__':
    main(sys.argv[1], limpar='--limpar' in sys.argv[2:])
