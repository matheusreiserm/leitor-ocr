"""Capa, orelhas, contracapa e folha de rosto recortadas do próprio escaneamento, no tamanho das páginas do livro.

Uso: py capas.py "<livro.pdf>" "<pasta do livro limpo>"
Lê   <pasta>/reconstrucao/config.json -> "capas": {"antes": [...], "depois": [...]}, cada item
     {"folha": 54, "caixa": [x0, y0, x1, y1], "nome": "capa"}, com a caixa em pixels da imagem já na posição
     de leitura (como preparar.imagem a devolve), e opcional "tons": true para página de texto (folha de
     rosto): papel levado a branco e cinza, como o miolo, e recebe a textura de papel no PDF "papel".
Grava <pasta>/capas/antes_NN_<nome>.png e depois_NN_<nome>.png, que o pdfs_livro.py põe antes e depois do miolo.
Capa e contracapa (proporção perto da página) preenchem a página; orelhas, mais estreitas, ficam inteiras e
centradas sobre fundo branco.
"""
import sys, json
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2, numpy as np, pymupdf
from preparar import fundo_branco

GIRA = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def imagem_cor(doc, i):
    ims = doc[i].get_images(full=True)
    vis = [im for im in ims if doc[i].get_image_rects(im[0])]
    x = doc.extract_image((vis or ims)[0][0])
    c = cv2.imdecode(np.frombuffer(x['image'], np.uint8), cv2.IMREAD_COLOR)
    g = GIRA.get(doc[i].rotation % 360)
    return cv2.rotate(c, g) if g is not None else c


def no_tamanho(img, W, H):
    h, w = img.shape[:2]
    if abs((w / h) / (W / H) - 1) <= 0.12:   # preenche e corta o excesso no centro
        e = max(W / w, H / h); r = cv2.resize(img, (round(w * e), round(h * e)), interpolation=cv2.INTER_AREA if e < 1 else cv2.INTER_CUBIC)
        y, x = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
        return r[y:y + H, x:x + W]
    e = min(W / w, H / h); r = cv2.resize(img, (round(w * e), round(h * e)), interpolation=cv2.INTER_AREA if e < 1 else cv2.INTER_CUBIC)
    out = np.full((H, W) + img.shape[2:], 255, np.uint8)
    y, x = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def main(pdf, pasta):
    pasta = Path(pasta); cfg = json.load(open(pasta / 'reconstrucao' / 'config.json', encoding='utf-8-sig'))
    spec = cfg.get('capas', {})
    pag = sorted((pasta / 'paginas').glob('*.png'))[1]
    H, W = cv2.imread(str(pag), cv2.IMREAD_GRAYSCALE).shape
    doc = pymupdf.open(pdf); out = pasta / 'capas'; out.mkdir(exist_ok=True)
    for f in out.glob('*.png'): f.unlink()
    for lugar in ('antes', 'depois'):
        for k, it in enumerate(spec.get(lugar, []), 1):
            x0, y0, x1, y1 = it['caixa']
            c = imagem_cor(doc, it['folha'] - 1)[y0:y1, x0:x1]
            if it.get('tons'):
                g = fundo_branco(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)); g[g > 200] = 255
                c = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            dst = out / f'{lugar}_{k:02d}_{it["nome"]}.png'
            cv2.imwrite(str(dst), no_tamanho(c, W, H))
            print(dst.name, f'(folha {it["folha"]}, {x1 - x0}×{y1 - y0} px -> {W}×{H})')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
