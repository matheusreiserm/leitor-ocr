"""PDF digital (texto vetorial, ex.: InDesign) já perfeito: só põe um fundo leve de papel por baixo e a capa.

Uso: py papel_pdf.py "<livro.pdf>" "<saida.pdf>" [--capa <imagem>] [--contracapa <imagem>] [--forca 0.5]
- O texto continua vetorial (nítido, arquivo pequeno); o papel é uma imagem leve desenhada atrás do conteúdo, uma
  para as páginas da esquerda e outra para as da direita (sombra da lombada do lado certo), reaproveitada em todas.
- --forca: quanto da textura entra (0 = branco, 1 = a do papel.py); padrão 0.5, de leve.
- marcas de corte da gráfica (traços nas bordas) são tiradas; --sem-tirar-marcas mantém.
- --capa / --contracapa: imagem que ocupa a primeira / a última página (que no PDF da gráfica vêm em branco).
"""
import sys, argparse
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
import cv2, numpy as np, pymupdf
import papel

DPI = 150


def textura(w_pt, h_pt, lado, forca, semente):
    W, H = int(w_pt * DPI / 72), int(h_pt * DPI / 72)
    t = papel.textura(H, W, semente, lado)
    t = 255 - (255 - t) * forca
    return cv2.imencode('.jpg', np.clip(t, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pdf'); ap.add_argument('saida')
    ap.add_argument('--capa'); ap.add_argument('--contracapa'); ap.add_argument('--forca', type=float, default=0.5)
    ap.add_argument('--sem-tirar-marcas', dest='marcas', action='store_false'); a = ap.parse_args()
    doc = pymupdf.open(a.pdf); n = doc.page_count
    capas = {0: a.capa, n - 1: a.contracapa}
    xref = {}
    for i, pg in enumerate(doc):
        if capas.get(i):
            r = pg.trimbox if pg.trimbox != pg.mediabox else pg.rect
            pg.set_cropbox(r); pg.insert_image(r, filename=capas[i], keep_proportion=False, overlay=True)
            continue
        if a.marcas:
            # marcas de corte da gráfica: traços vetoriais que encostam nas bordas; tira só o desenho que toca uma faixa
            # de 25 pt junto à borda (texto e imagens ficam), sem mudar o tamanho da página
            m = pg.rect; f = 25
            for r in (pymupdf.Rect(m.x0, m.y0, m.x1, m.y0 + f), pymupdf.Rect(m.x0, m.y1 - f, m.x1, m.y1),
                      pymupdf.Rect(m.x0, m.y0, m.x0 + f, m.y1), pymupdf.Rect(m.x1 - f, m.y0, m.x1, m.y1)):
                pg.add_redact_annot(r, fill=False)
            pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                                text=pymupdf.PDF_REDACT_TEXT_NONE)
        lado = 'dir' if (i + 1) % 2 == 1 else 'esq'
        chave = (lado, round(pg.mediabox.width), round(pg.mediabox.height))
        if chave not in xref:
            xref[chave] = pg.insert_image(pg.mediabox, stream=textura(pg.mediabox.width, pg.mediabox.height, lado, a.forca, 7 + len(xref)), overlay=False)
        else:
            pg.insert_image(pg.mediabox, xref=xref[chave], overlay=False)
    doc.save(a.saida, garbage=3, deflate=True)
    print(f'{a.saida}: {n} páginas, {_o.path.getsize(a.saida) / 1e6:.1f} MB' + ('' if a.capa else ' (sem capa)'))


if __name__ == '__main__':
    main()
