"""Versão final de um livro já revisado: reconstruções aprovadas desenhadas sobre as páginas da limpeza nova.

Uso: py -I -X utf8 aplicar_sobre_limpa.py "<pasta do livro limpo>" "<páginas-base antigas>" "<páginas-base limpas>"

Igual ao aplicar.py do reconstrucao-livros, com duas diferenças:
- a geometria de cada palavra (onde apagar, onde desenhar) é medida na página-base ANTIGA, a mesma sobre a
  qual a revisão foi feita; a limpeza nova apaga manchas, e manchas a menos mudariam essa medida (a área
  apagada encolhe e deixa restos ao lado da palavra). As palavras redesenhadas saem idênticas às aprovadas;
- grava as 77 páginas, não só as que têm reconstrução (o aplicar.py deixa as outras como estavam).
"""
import sys, json, collections
from pathlib import Path
FER = Path(__file__).resolve().parent.parent / 'ferramentas_reconstrucao'; sys.path.insert(0, str(FER))  # repositório reconstrucao-livros, ao lado
import uniio  # noqa: caminhos com acento
import cv2
from glifos import Motor
from casos import geometria, num_pagina


def main(pasta, base_antiga, base_nova):
    pasta, base_antiga, base_nova = Path(pasta), Path(base_antiga), Path(base_nova)
    rd = pasta / 'reconstrucao'
    props = {str(p['id']): p for p in json.load(open(rd / 'propostas.json', encoding='utf-8'))}
    dec = json.load(open(rd / 'decisoes.json', encoding='utf-8'))
    ocr = json.load(open(pasta / 'ocr.json', encoding='utf-8'))
    motor = Motor(pasta)
    por_pag = collections.defaultdict(list)
    for k, d in dec.items():
        if d['acao'] in ('aceitar', 'corrigir'): por_pag[props[k]['pagina']].append((props[k], d['texto']))
    nomes = sorted(f.name for f in base_nova.glob('*.png'))
    assert nomes == sorted(f.name for f in base_antiga.glob('*.png')), 'as duas pastas-base não têm as mesmas páginas'
    log = []
    for nome in nomes:
        antiga = cv2.imread(str(base_antiga / nome), cv2.IMREAD_GRAYSCALE)   # mede a geometria
        nova = cv2.imread(str(base_nova / nome), cv2.IMREAD_GRAYSCALE)       # recebe o desenho
        assert antiga.shape == nova.shape, nome
        for p, texto in por_pag.get(nome, []):
            g = geometria(motor, ocr[nome], p, texto, num_pagina(nome), antiga)
            args = (g['texto'], g['baseline'], g['x0'], g['x1'], g['ancora'], g['apagar'])
            antiga = motor.renderizar(antiga, *args)
            nova = motor.renderizar(nova, *args)
            log.append(f'página {num_pagina(nome):3d} · caso {p["id"]:4d} · "{p["texto"]}" → "{texto}"   | {p["linha"]}')
        cv2.imwrite(str(pasta / 'paginas' / nome), nova)
    (pasta / 'reconstrucoes.txt').write_text('Palavras reconstruídas (aprovadas na revisão)\n\n' + '\n'.join(log) + '\n', encoding='utf-8')
    print(f'{len(nomes)} páginas gravadas; {len(log)} palavras reconstruídas em {len(por_pag)} páginas.')


if __name__ == '__main__':
    main(*sys.argv[1:4])
