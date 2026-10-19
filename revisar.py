"""Revisor de reconstruções (terminal + janela de prévia).

Uso:  py revisar.py "<pasta do livro limpo>"
Lê   <pasta>/reconstrucao/propostas.json
Grava <pasta>/reconstrucao/decisoes.json (a cada resposta; pode parar e continuar depois)
"""
import sys, json, threading, queue, os
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); import uniio  # noqa: caminhos com acento
_s.stdout.reconfigure(encoding='utf-8'); _s.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).parent))
from glifos import Motor
from casos import previa

R, N, B, A, V, C = '\033[91m', '\033[0m', '\033[1m', '\033[93m', '\033[92m', '\033[96m'
os.system('')  # habilita cores ANSI no console do Windows


def entrada(prompt, janela):
    """input() numa thread enquanto a janela de prévia continua respondendo."""
    q = queue.Queue()
    def ler():
        try: q.put(input(prompt))
        except (EOFError, KeyboardInterrupt): q.put('q')
    threading.Thread(target=ler, daemon=True).start()
    while True:
        try: return q.get_nowait()
        except queue.Empty: cv2.waitKey(40)


def main(pasta):
    pasta = Path(pasta); rd = pasta / 'reconstrucao'
    props = json.load(open(rd / 'propostas.json', encoding='utf-8'))
    ocr = json.load(open(pasta / 'ocr.json', encoding='utf-8'))
    motor = Motor(pasta)
    fdec = rd / 'decisoes.json'
    dec = json.load(open(fdec, encoding='utf-8')) if fdec.exists() else {}
    janela = 'Previa - ANTES / DEPOIS'
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    pend = [p for p in props if str(p['id']) not in dec]
    print(f'\n{B}{len(props)} casos, {len(props) - len(pend)} já decididos, {len(pend)} pendentes.{N}')
    print('Comandos: [Enter] aceitar a proposta | n = manter como está | digite a palavra correta | p = pular | v = voltar | q = sair')
    print('          A = aceitar de uma vez todos os pendentes de certeza alta (e seguir com os outros)\n')
    k = 0
    while k < len(pend):
        p = pend[k]
        img, _, _ = previa(pasta, motor, ocr, p, p['proposta'])
        cv2.imshow(janela, img); cv2.resizeWindow(janela, min(1400, img.shape[1]), min(500, img.shape[0])); cv2.waitKey(1)
        linha = p['linha'].replace(p['texto'], f'{R}{B}[{p["texto"]}]{N}', 1)
        print(f'{C}── caso {p["id"]} ({k + 1}/{len(pend)}) · página {int(p["pagina"][:3])} ─────────────{N}')
        if p.get('antes'): print(f'   {p["antes"]}')
        print(f'   {linha}')
        if p.get('depois'): print(f'   {p["depois"]}')
        print(f'   OCR leu: {A}{p["texto"]}{N}   →   proposta: {V}{B}{p["proposta"]}{N}   (certeza: {p.get("certeza", "?")})')
        if p.get('nota'): print(f'   nota: {p["nota"]}')
        r = entrada('   > ', janela).strip()
        if r == '': dec[str(p['id'])] = dict(acao='aceitar', texto=p['proposta'])
        elif r.lower() == 'n': dec[str(p['id'])] = dict(acao='manter')
        elif r.lower() == 'p': k += 1; continue
        elif r.lower() == 'v': k = max(0, k - 1); dec.pop(str(pend[k]['id']), None); continue
        elif r.lower() == 'q': break
        elif r == 'A':
            altas = [q for q in pend[k:] if str(q['id']) not in dec and str(q.get('certeza', '')).startswith('alta')]
            for q in altas: dec[str(q['id'])] = dict(acao='aceitar', texto=q['proposta'])
            json.dump(dec, open(fdec, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            pend = [q for q in props if str(q['id']) not in dec]; k = 0
            print(f'   {V}{len(altas)} casos de certeza alta aceitos.{N} Faltam {len(pend)}.\n')
            continue
        else:
            while True:
                img, _, _ = previa(pasta, motor, ocr, p, r)
                cv2.imshow(janela, img); cv2.waitKey(1)
                falt = motor.faltando(r)
                if falt: print(f'   {A}aviso: o livro não tem amostra limpa de {falt}; ficará um espaço no lugar.{N}')
                r2 = entrada(f'   aplicar "{r}"? [Enter] sim | digite outra palavra | n = manter original > ', janela).strip()
                if r2 == '': dec[str(p['id'])] = dict(acao='corrigir', texto=r); break
                if r2.lower() == 'n': dec[str(p['id'])] = dict(acao='manter'); break
                r = r2
        json.dump(dec, open(fdec, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        k += 1
    cv2.destroyAllWindows()
    ac = sum(d['acao'] != 'manter' for d in dec.values())
    print(f'\n{B}Decididos: {len(dec)}/{len(props)} · a reconstruir: {ac}.{N} Decisões salvas em {fdec}')
    print('Quando terminar, avise o Claude (ou rode aplicar.ps1) para aplicar as mudanças e refazer os PDFs.')


if __name__ == '__main__':
    main(sys.argv[1])
