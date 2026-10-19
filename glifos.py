"""Motor de reconstrução: monta palavras com letras recortadas do próprio livro (sem fonte externa)."""
import json, pickle, unicodedata
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
from pathlib import Path
import cv2, numpy as np

BASE = set('acemnorsuvwxzáéíóúñ')       # letras apoiadas na linha de base, sem descendente
INK = 140


def _tight(gray, x0, y0, x1, y1, pad=3):
    H, W = gray.shape
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = gray[Y0:Y1, X0:X1]
    ys, xs = np.where(crop < INK)
    if len(xs) == 0: return None
    return X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1


def _dobra(t):
    return t.replace('ç', 'c').replace('Ç', 'C').replace('ü', 'u').replace('ú', 'u').replace('Ü', 'U')


def _por_pecas(gray, w, t):
    """Segmenta a palavra em peças de tinta (juntando acentos/pingos à letra de baixo) e, se o número
    de peças == número de letras, devolve (letra, imagem, dy em relação à linha de base) em ordem."""
    X, Y, W, H = w['x'], w['y'], w['w'], w['h']
    p = int(H * 0.5)
    Y0, Y1 = max(0, Y - p), min(gray.shape[0], Y + H + p)
    reg = (gray[Y0:Y1, X:X + W] < 200).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(reg, 8)
    pecas = [[i] for i in range(1, k) if st[i][4] >= 6]
    # junta peças cujo intervalo em x está contido em outra (acentos, pingos, cedilhas soltas)
    pecas.sort(key=lambda g: st[g[0]][0])
    juntas = []
    for g in pecas:
        x0, x1 = st[g[0]][0], st[g[0]][0] + st[g[0]][2]
        if juntas:
            a0 = min(st[i][0] for i in juntas[-1]); a1 = max(st[i][0] + st[i][2] for i in juntas[-1])
            sob = min(x1, a1) - max(x0, a0)
            if sob > 0.5 * min(x1 - x0, a1 - a0): juntas[-1] += g; continue
        juntas.append(g)
    if len(juntas) != len(t): return []
    base = [max(st[i][1] + st[i][3] for i in g) for c, g in zip(t, juntas) if c in BASE]
    if not base: return []
    bl = int(np.median(base))
    out = []
    for c, g in zip(t, juntas):
        m = np.isin(lab, g); ys, xs = np.where(m)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        img = np.full((y1 - y0, x1 - x0), 255, np.uint8)
        sub = gray[Y0 + y0:Y0 + y1, X + x0:X + x1]; mm = m[y0:y1, x0:x1]; img[mm] = sub[mm]
        out.append((c, img, int(y1 - bl)))
    return out


def construir(pasta):
    """Lê ocr.json + páginas e monta a biblioteca de glifos do livro (glifos.pkl)."""
    pasta = Path(pasta); ocr = json.load(open(pasta / 'ocr.json', encoding='utf-8'))
    amostras, gaps, xh = {}, [], []
    # o OCR não reconhece ç/ü (lê c/g e ú): palavras com essas letras vindas das propostas servem de gabarito
    gabarito = {}
    fp = pasta / 'reconstrucao' / 'propostas.json'
    if fp.exists():
        for p in json.load(open(fp, encoding='utf-8')):
            for v in p['proposta'].split():
                if any(c in 'çÇüÜ' for c in v):
                    for parte in [v] + [v[:k] for k in range(4, len(v))]:  # também prefixos (contigüidad ⊂ contigüidades)
                        gabarito[_dobra(parte)] = parte
    for nome, pg in ocr.items():
        gray = cv2.imread(str(pasta / 'paginas' / nome), cv2.IMREAD_GRAYSCALE)
        chars = pg['chars']
        cx = np.array([(c[1] + c[3]) / 2 for c in chars]); cy = np.array([(c[2] + c[4]) / 2 for c in chars])
        for w in pg['words']:
            t = w['t'].strip('¿?¡!.,;:…"“”«»()[]—–\'’‘*')  # pontuação colada não impede usar as letras
            lido = t
            if _dobra(t) in gabarito and t != gabarito[_dobra(t)]: t = gabarito[_dobra(t)]
            raras = any(c in 'çÇüÜñÑ' for c in t)
            if any(c in 'çÇüÜ' for c in t) and t != lido:
                for ch, g, dy in _por_pecas(gray, w, t):
                    if ch in 'çÇüÜ': amostras.setdefault(ch, []).append(dict(img=g, dy=dy, w=g.shape[1], h=g.shape[0]))
            if w['conf'] < (70 if raras else 93) or not t.isalpha() or len(t) < 2 or not len(chars): continue
            sel = np.where((cx >= w['x'] - 2) & (cx <= w['x'] + w['w'] + 2) & (cy >= w['y'] - 2) & (cy <= w['y'] + w['h'] + 2))[0]
            cs = [chars[k] for k in sel if chars[k][0].isalpha()]
            if len(cs) != len(t) or ''.join(c[0] for c in cs) != lido: continue
            boxes = [_tight(gray, c[1], c[2], c[3], c[4], pad=1) for c in cs]
            if any(b is None for b in boxes): continue
            base = [b[3] for ch, b in zip(t, boxes) if ch.lower() in BASE and ch.islower()]
            if not base: continue
            bl = int(np.median(base))
            xs_ = [b for ch, b in zip(t, boxes) if ch in 'acemnorsuvwxz']
            if xs_: xh.append(np.median([bl - b[1] for b in xs_]))
            for a, b in zip(boxes, boxes[1:]): gaps.append(b[0] - a[2])
            for k_, (ch, b) in enumerate(zip(t, boxes)):
                # ligaduras (fi, fl, ff): o OCR separa, mas a tinta é uma peça só -> não servem de amostra
                if (ch == 'f' and k_ + 1 < len(t) and t[k_ + 1] in 'filt') or (ch in 'filt' and k_ > 0 and t[k_ - 1] == 'f'): continue
                # recorte pelas peças de tinta conectadas cujo centro cai na caixa da letra
                # (pega o gancho do "f", o pingo do "i", acentos; exclui tinta das vizinhas)
                bx0, by0, bx1, by1 = b
                pw = bx1 - bx0
                X0, Y0 = max(0, bx0 - pw), max(0, by0 - pw)
                X1, Y1 = min(gray.shape[1], bx1 + pw), min(gray.shape[0], by1 + pw)
                reg = (gray[Y0:Y1, X0:X1] < 200).astype(np.uint8)
                k, lab, st, cen = cv2.connectedComponentsWithStats(reg, 8)
                sel_ = [i for i in range(1, k) if bx0 <= X0 + cen[i][0] <= bx1 and st[i][4] >= 3
                        and Y0 + st[i][1] < by1 and Y0 + st[i][1] + st[i][3] > by0 - 0.3 * (by1 - by0)]  # na altura da letra
                if not sel_: continue
                if any(st[i][0] == 0 or st[i][0] + st[i][2] >= reg.shape[1] for i in sel_): continue  # peça grande demais (letras coladas)
                m_ = np.isin(lab, sel_)
                ys, xs = np.where(m_)
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                if (x1 - x0) > 1.9 * max(pw, 4): continue
                g = np.full((y1 - y0, x1 - x0), 255, np.uint8)
                sub = gray[Y0 + y0:Y0 + y1, X0 + x0:X0 + x1]
                mm = m_[y0:y1, x0:x1]
                g[mm] = sub[mm]
                amostras.setdefault(ch, []).append(dict(img=g, dy=Y0 + y1 - bl, w=x1 - x0, h=y1 - y0))
    lib, tinta = {}, {}
    for ch, L in amostras.items():
        for a in L: a['ink'] = int((a['img'] < INK).sum())
        if len(L) >= 6:  # descarta caixas largas demais (OCR juntou a letra vizinha)
            lim = np.percentile([a['w'] for a in L], 40) * 1.25
            L = [a for a in L if a['w'] <= lim] or L
        mw, mh, md = (np.median([a[k] for a in L]) for k in ('w', 'h', 'dy'))
        tinta[ch] = float(np.median([a['ink'] for a in L]))
        # candidatos de tamanho típico; entre eles, o mais parecido com a média de todas as ocorrências (o "medoide")
        L.sort(key=lambda a: abs(a['w'] - mw) + abs(a['h'] - mh) + abs(a['dy'] - md))
        top = L[:200]
        W_, H_ = max(1, int(mw)), max(1, int(mh))
        R = np.stack([cv2.resize(a['img'], (W_, H_), interpolation=cv2.INTER_AREA).astype(np.float32) for a in top])
        media = R.mean(0)
        dist = ((R - media) ** 2).mean(axis=(1, 2))
        ordem = np.argsort(dist)
        lib[ch] = [top[k] for k in ordem[:5]]
    meta = dict(gap=float(np.median(gaps)) if gaps else 3.0, xh=float(np.median(xh)) if xh else 30.0,
                n={k: len(v) for k, v in amostras.items()}, tinta=tinta)
    pickle.dump(dict(lib=lib, meta=meta), open(pasta / 'glifos.pkl', 'wb'))
    return lib, meta


class Motor:
    def __init__(self, pasta):
        d = pickle.load(open(Path(pasta) / 'glifos.pkl', 'rb'))
        self.lib, self.meta = d['lib'], d['meta']
        # opções por livro (ex.: {"geometria": 2}); sem o arquivo, o comportamento original
        fc = Path(pasta) / 'reconstrucao' / 'config.json'
        self.cfg = json.load(open(fc, encoding='utf-8-sig')) if fc.exists() else {}

    def glifo(self, ch):
        if ch in self.lib: return self.lib[ch][0]
        # variante sem acento + acento emprestado não é feito: avisa
        return None

    def faltando(self, texto):
        return sorted({c for c in texto if c != ' ' and c not in self.lib})

    def largura(self, texto):
        gs = [self.glifo(c) for c in texto]
        sp = self.meta['xh'] * 0.55
        return sum((g['w'] if g else sp) for g in gs) + self.meta['gap'] * (len(texto) - 1)

    def renderizar(self, page, texto, baseline, x0=None, x1=None, ancora='esq', apagar=None):
        """Desenha `texto` na página (cópia). ancora: 'esq' (começa em x0), 'dir' (termina em x1), 'ajustar' (entre x0 e x1).
        apagar: caixa (x0,y0,x1,y1) a limpar antes (restos da palavra original)."""
        out = page.copy()
        if isinstance(apagar, np.ndarray):
            out[apagar] = 255
        elif apagar is not None:
            a0, b0, a1, b1 = [int(v) for v in apagar]
            out[b0:b1, a0:a1] = 255
        nat = self.largura(texto)
        gap = self.meta['gap']
        if ancora == 'ajustar' and x0 is not None and x1 is not None and len(texto) > 1:
            gap = gap + ((x1 - x0) - nat) / (len(texto) - 1); x = x0
        elif ancora == 'dir': x = x1 - nat
        else: x = x0
        sp = self.meta['xh'] * 0.55
        for c in texto:
            g = self.glifo(c)
            if g is None: x += sp + gap; continue
            xi, yi = int(round(x)), int(round(baseline + g['dy'] - g['h']))
            H, W = out.shape
            if 0 <= yi and yi + g['h'] <= H and 0 <= xi and xi + g['w'] <= W:
                reg = out[yi:yi + g['h'], xi:xi + g['w']]
                np.minimum(reg, g['img'], out=reg)
            x += g['w'] + gap
        return out
