#!/usr/bin/env python3
"""Avalia OCR vs transcrições humanas em .txt com o mesmo nome-base.

Exemplo: py -I -X utf8 avaliar_gabarito.py --referencias gabarito_humano --hipoteses tess_saida --saida avaliacao
Usa apenas biblioteca padrão. Não remove diacríticos, e ignora pontuação e caixa.
"""
from __future__ import annotations
import argparse, csv, json, re, unicodedata
from pathlib import Path


def palavras(texto: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", texto.casefold(), flags=re.UNICODE)


def distancia_edicao(ref: list[str], hyp: list[str]):
    """Retorna custo mínimo e um alinhamento em nível de palavras/caracteres."""
    m, n = len(ref), len(hyp)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1): d[i][0] = i
    for j in range(1, n + 1): d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = m, n
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = 'igual' if ref[i - 1] == hyp[j - 1] else 'substituicao'
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(('exclusao', ref[i - 1], ''))
            i -= 1
        else:
            ops.append(('insercao', '', hyp[j - 1]))
            j -= 1
    ops.reverse()
    return d[m][n], ops


def sem_marcas(texto: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')


def avaliar(ref_texto: str, hip_texto: str) -> tuple[dict, list[dict]]:
    ref = palavras(ref_texto); hip = palavras(hip_texto)
    if not ref: raise ValueError('Referência sem palavras reconhecíveis.')
    dist_w, ops = distancia_edicao(ref, hip)
    # CER lexical: sobre caracteres das palavras unidos por espaços, sem pontuação.
    ref_chars = list(' '.join(ref))
    hip_chars = list(' '.join(hip))
    dist_c, _ = distancia_edicao(ref_chars, hip_chars)
    erros = []
    for i, (op, r, h) in enumerate(ops, 1):
        if op == 'igual': continue
        somente_diacriticos = op == 'substituicao' and r != h and sem_marcas(r) == sem_marcas(h)
        erros.append({'posicao_alinhada': i, 'tipo': op, 'referencia': r, 'ocr': h,
                      'somente_diacriticos': somente_diacriticos})
    resultado = {'palavras_referencia': len(ref), 'palavras_ocr': len(hip),
                 'erros_palavra': dist_w, 'wer_pct': 100 * dist_w / len(ref),
                 'caracteres_referencia': len(ref_chars), 'erros_caractere': dist_c,
                 'cer_pct': 100 * dist_c / len(ref_chars),
                 'substituicoes_so_diacriticos': sum(bool(e['somente_diacriticos']) for e in erros)}
    return resultado, erros


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--referencias', type=Path, required=True)
    p.add_argument('--hipoteses', type=Path, required=True)
    p.add_argument('--saida', type=Path, required=True)
    args = p.parse_args()
    ref_files = {f.stem: f for f in args.referencias.glob('*.txt')}
    hip_files = {f.stem: f for f in args.hipoteses.glob('*.txt')}
    pares = sorted(ref_files.keys() & hip_files.keys())
    if not pares: raise SystemExit('Nenhum par de .txt com o mesmo nome-base encontrado.')
    args.saida.mkdir(parents=True, exist_ok=True)
    paginas, erros = [], []
    for nome in pares:
        a, b = (ref_files[nome].read_text(encoding='utf-8'), hip_files[nome].read_text(encoding='utf-8'))
        metricas, diferencas = avaliar(a, b)
        paginas.append({'pagina': nome, **metricas})
        erros += [{'pagina': nome, **row} for row in diferencas]
    totals = {k: sum(p[k] for p in paginas) for k in ['palavras_referencia','palavras_ocr',
               'erros_palavra','caracteres_referencia','erros_caractere','substituicoes_so_diacriticos']}
    totals['wer_pct'] = 100*totals['erros_palavra']/totals['palavras_referencia']
    totals['cer_pct'] = 100*totals['erros_caractere']/totals['caracteres_referencia']
    (args.saida/'resumo.json').write_text(json.dumps({'totais': totals,'paginas':paginas,
        'sem_referencia': sorted(hip_files.keys()-ref_files.keys()),
        'sem_ocr': sorted(ref_files.keys()-hip_files.keys())}, ensure_ascii=False,indent=2), encoding='utf-8')
    for nome, linhas in [('paginas.csv', paginas), ('divergencias.csv', erros)]:
        if linhas:
            with (args.saida/nome).open('w', newline='', encoding='utf-8-sig') as fp:
                wr=csv.DictWriter(fp, fieldnames=list(linhas[0]));wr.writeheader();wr.writerows(linhas)
    print(f'{len(pares)} página(s) pareada(s): WER={totals["wer_pct"]:.2f}% '
          f'CER={totals["cer_pct"]:.2f}%  '
          f'substituições só de diacríticos={totals["substituicoes_so_diacriticos"]}')
    print(f'Resultados em: {args.saida}')

if __name__ == '__main__':
    main()
