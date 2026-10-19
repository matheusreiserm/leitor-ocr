"""Aplica as reconstruções aprovadas nas páginas e registra tudo em reconstrucoes.txt.

Uso: py aplicar.py "<pasta do livro limpo>"
As páginas originais (antes da reconstrução) ficam guardadas em paginas_sem_reconstrucao/.
"""
import sys, json, shutil, collections
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).parent))
from glifos import Motor
from casos import geometria, num_pagina


def main(pasta):
    pasta = Path(pasta); rd = pasta / 'reconstrucao'
    props = {str(p['id']): p for p in json.load(open(rd / 'propostas.json', encoding='utf-8'))}
    dec = json.load(open(rd / 'decisoes.json', encoding='utf-8'))
    ocr = json.load(open(pasta / 'ocr.json', encoding='utf-8'))
    motor = Motor(pasta)
    bak = pasta / 'paginas_sem_reconstrucao'
    if not bak.exists(): shutil.copytree(pasta / 'paginas', bak)
    por_pag = collections.defaultdict(list)
    for k, d in dec.items():
        if d['acao'] in ('aceitar', 'corrigir'): por_pag[props[k]['pagina']].append((props[k], d['texto']))
    log = []
    for nome, itens in sorted(por_pag.items()):
        page = cv2.imread(str(bak / nome), cv2.IMREAD_GRAYSCALE)  # sempre parte da versão sem reconstrução
        for p, texto in itens:
            g = geometria(motor, ocr[nome], p, texto, num_pagina(nome), page)
            page = motor.renderizar(page, g['texto'], g['baseline'], g['x0'], g['x1'], g['ancora'], g['apagar'])
            log.append(f'página {num_pagina(nome):3d} · caso {p["id"]:4d} · "{p["texto"]}" → "{texto}"   | {p["linha"]}')
        cv2.imwrite(str(pasta / 'paginas' / nome), page)
    (pasta / 'reconstrucoes.txt').write_text('Palavras reconstruídas (aprovadas na revisão)\n\n' + '\n'.join(log) + '\n', encoding='utf-8')
    print(f'{len(log)} palavras reconstruídas em {len(por_pag)} páginas. Registro: {pasta / "reconstrucoes.txt"}')


if __name__ == '__main__':
    main(sys.argv[1])
