"""Livro em páginas duplas já legível: só corta cada folha em duas páginas, sem mexer na imagem, e põe por baixo
o texto do OCR (invisível), para buscar e selecionar.

Uso: py fatiar.py "<livro.pdf>" "<pasta do livro>" "<título>"
Lê <pasta>/reconstrucao/config.json: "idioma", "ignorar" (folhas ou metades fora do miolo) e "capas" (recortes
de capa, orelhas e contracapa, postos antes e depois do miolo, também sem tratamento).
Cada página é o recorte do escaneamento original, em cor, sem clarear nem limpar; só endireitada e recortada
no mesmo tamanho, com a mancha no mesmo lugar ("endireitar": false no config.json desliga). Orelhas ficam como são.
Grava <pasta>/<título>.pdf e <pasta>/paginas_originais/ (as páginas recortadas, para conferência).
"""
import sys, json, subprocess, tempfile
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pymupdf
import idioma
from preparar import fundo_branco, lombada_vao
from capas import imagem_cor
from ocr import TESS, TESSDATA

DPI = 300


def texto_invisivel(img, branquear=True):
    """PDF de uma página só com o texto do Tesseract (sem imagem), no tamanho da imagem a 300 dpi."""
    with tempfile.TemporaryDirectory() as td:
        # o OCR lê uma cópia com o papel a branco (a sombra da lombada atrapalha); a imagem do livro não muda
        # (só para escaneamento com sombra/papel escuro: em página já limpa, branquear afina os acentos e o OCR os perde)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        src = Path(td) / 'p.png'; cv2.imwrite(str(src), fundo_branco(g) if branquear else g)
        subprocess.run([TESS, str(src), str(Path(td) / 'p'), '--tessdata-dir', str(TESSDATA), '-l', idioma.tesseract(),
                        '--dpi', str(DPI), '-c', 'textonly_pdf=1', 'pdf'], check=True, capture_output=True,
                       env=dict(_o.environ, OMP_THREAD_LIMIT='1'))
        return (Path(td) / 'p.pdf').read_bytes()


MARGEM = 190   # px a 300 dpi em volta da mancha (cabeçalho a número de página)


def tirar_fundo(c, borda=10):
    """Escaneamento com a tampa aberta: o papel é a grande área clara sobre o fundo preto. Fora dele (e numa faixa
    de alguns px na beira, onde fica a sombra da borda da folha) vira cor do papel; dentro, nada muda."""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    claro = (g > 90).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(claro, 8)
    if k < 2: return c
    i = 1 + int(np.argmax(st[1:, 4]))
    cont, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    papel_m = np.zeros(g.shape, np.uint8); cv2.drawContours(papel_m, cont, -1, 1, -1)   # com o texto (os "buracos") dentro
    papel_m = cv2.erode(papel_m, np.ones((3, 3), np.uint8), iterations=borda).astype(bool)
    if papel_m.all(): return c
    cor = np.median(c[papel_m & (g > 150)], axis=0) if (papel_m & (g > 150)).any() else np.array([245, 245, 245])
    out = c.copy(); out[~papel_m] = cor.astype(np.uint8)
    return out


def endireitar_e_igualar(paginas, centralizar=(), fundo_escuro=False):
    """Gira cada página pela inclinação das linhas e recorta todas no mesmo tamanho, com a mancha no mesmo lugar:
    na horizontal pela mancha (ou, em página sem mancha cheia, pela lombada, como as do mesmo lado), na vertical
    pelo número da página (ou, sem ele, como as do mesmo lado). Onde o escaneamento não chega (margem interna),
    completa com a cor do papel da própria página. A imagem em si não é tratada."""
    from preparar import inclinacao, analisar
    med = []
    for nome, c in paginas:
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY); b = np.where(fundo_branco(g) < 180, 0, 255).astype(np.uint8)
        a = inclinacao(b, 21); a += inclinacao(_girar(b, a, 255), 21)
        r = analisar(_girar(b, a, 255), 21)
        cg = _girar(c, a, None)
        if fundo_escuro: cg = tirar_fundo(cg)
        med.append(dict(nome=nome, lado=nome[-3:], c=cg, ang=a, caixa=r['caixa'], folio=r['folio']))
    cheias = [m for m in med if m['caixa'] and m['folio'] is not None]
    tw = float(np.median([m['caixa'][2] - m['caixa'][0] for m in cheias]))
    W = int(tw + 2 * MARGEM)
    alto = float(np.median([m['folio'] - m['caixa'][1] for m in cheias]))   # do topo da mancha ao número
    H = int(alto + 2 * MARGEM + 25)
    for m in med:
        w = m['c'].shape[1]; cx = m['caixa']
        cheia = cx and abs((cx[2] - cx[0]) - tw) < 0.1 * tw
        if cheia: m['x0'] = cx[0] - MARGEM
        if m['folio'] is not None and cx: m['y0'] = m['folio'] + MARGEM + 25 - H
        m['lomb'] = (m['x0'] if m['lado'] == 'dir' else w - (m['x0'] + W)) if cheia else None
    for lado in ('esq', 'dir'):
        L_ = [m for m in med if m['lado'] == lado]
        lomb = float(np.median([m['lomb'] for m in L_ if m['lomb'] is not None]))
        y0 = float(np.median([m['y0'] for m in L_ if 'y0' in m]))
        for m in L_:
            if m['nome'] in centralizar:   # folha fora do padrão (ex.: rosto ao lado da orelha): janela no meio da página
                h_, w_ = m['c'].shape[:2]; m['x0'], m['y0'] = (w_ - W) / 2, (h_ - H) / 2; continue
            if 'x0' not in m: m['x0'] = lomb if lado == 'dir' else m['c'].shape[1] - lomb - W
            m.setdefault('y0', y0)
    out = []
    for m in med:
        c = m['c']; h, w = c.shape[:2]
        papel = np.median(c[int(h * 0.3):int(h * 0.7), :][np.median(c[int(h * 0.3):int(h * 0.7), :], axis=2) > 150], axis=0) \
            if c.size else np.array([235, 240, 245])
        tela = np.empty((H, W, 3), np.uint8); tela[:] = papel.astype(np.uint8)
        x0, y0 = int(round(m['x0'])), int(round(m['y0']))
        sx0, sy0 = max(0, x0), max(0, y0); sx1, sy1 = min(w, x0 + W), min(h, y0 + H)
        if sx1 > sx0 and sy1 > sy0: tela[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = c[sy0:sy1, sx0:sx1]
        out.append((m['nome'], tela))
    angs = [abs(m['ang']) for m in med]
    print(f'inclinação corrigida: mediana {np.median(angs):.2f}°, máxima {max(angs):.2f}° · páginas de {W} × {H} px')
    return out, (W, H)


def _girar(img, ang, borda):
    if abs(ang) < 0.05: return img
    h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    if borda is None: return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=borda)
    return np.where(r < 128, 0, 255).astype(np.uint8) if img.ndim == 2 else r


def main(pdf, pasta, titulo):
    pasta = Path(pasta); idioma.definir(pasta)
    cfg = json.load(open(pasta / 'reconstrucao' / 'config.json', encoding='utf-8-sig'))
    ignorar = {(int(f), l) for f, l in cfg.get('ignorar', [])}
    doc = pymupdf.open(pdf)
    paginas = []
    for i in range(len(doc)):
        if (i + 1, '*') in ignorar: continue
        c = imagem_cor(doc, i)
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        L = lombada_vao(np.where(fundo_branco(g) < 180, 0, 255).astype(np.uint8))   # só para achar onde cortar
        for lado, m in (('esq', c[:, :L]), ('dir', c[:, L:])):
            if (i + 1, lado) not in ignorar: paginas.append((f'{i + 1:02d}_{lado}', m))
    tam = None
    if cfg.get('endireitar', True): paginas, tam = endireitar_e_igualar(paginas, set(cfg.get('centralizar', [])), cfg.get('fundo_escuro', False))
    # páginas reconstruídas à mão (ex.: fim de linha entortado na lombada): {"098_49_esq": "reconstrucao/x.jpg"}
    # o nome é o da página recortada, sem o número de ordem; a imagem substitui o recorte, já no tamanho final
    subst = cfg.get('substituir', {})
    paginas = [(n, cv2.imdecode(np.fromfile(str(pasta / subst[n]), np.uint8), cv2.IMREAD_COLOR) if n in subst else im) for n, im in paginas]
    capas = cfg.get('capas', {})
    def recortes(lugar):
        from capas import no_tamanho
        out = []
        for k, it in enumerate(capas.get(lugar, []), 1):
            im = imagem_cor(doc, it['folha'] - 1)[it['caixa'][1]:it['caixa'][3], it['caixa'][0]:it['caixa'][2]]
            if tam and 'orelha' not in it['nome']: im = no_tamanho(im, *tam)   # capa e contracapa no tamanho das páginas
            out.append((f'{lugar}_{k}_{it["nome"]}', im))
        return out
    seq = recortes('antes') + paginas + recortes('depois')
    po = pasta / 'paginas_originais'; po.mkdir(exist_ok=True)
    for f in po.glob('*.jpg'): f.unlink()
    with ThreadPoolExecutor(max(1, (_o.cpu_count() or 2) - 1)) as ex:
        textos = list(ex.map(lambda nm: None if 'capa' in nm[0] or 'orelha' in nm[0] else texto_invisivel(nm[1]), seq))
    out = pymupdf.open()
    for k, ((nome, img), txt) in enumerate(zip(seq, textos), 1):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (po / f'{k:03d}_{nome}.jpg').write_bytes(buf.tobytes())
        h, w = img.shape[:2]; pg = out.new_page(width=w * 72 / DPI, height=h * 72 / DPI)
        pg.insert_image(pg.rect, stream=buf.tobytes())
        if txt:
            t = pymupdf.open('pdf', txt); pg.show_pdf_page(pg.rect, t, 0)
    dst = pasta / f'{titulo}.pdf'; out.save(str(dst), garbage=3, deflate=True)
    print(f'{dst.name}: {len(seq)} páginas ({len(paginas)} do miolo), {dst.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main(*sys.argv[1:4])
