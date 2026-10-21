import json, gzip, re, sys, collections
lex = json.load(gzip.open(sys.argv[1], 'rt', encoding='utf-8'))
ESP = set('abcdefghijklmnopqrstuvwxyzáéíóúüñ')
def palavras(txt):
    return [w for w in re.findall(r"[^\W\d_]+", txt.lower())]
def medir(nome, ws):
    ws = [w for w in ws if len(w) > 1]
    fora = [w for w in ws if w not in lex]
    estranho = [w for w in ws if set(w) - ESP]
    c = collections.Counter(fora)
    print(f"{nome:28} palavras={len(ws):6}  fora do léxico={len(fora):5} ({100*len(fora)/len(ws):.1f}%)  letras não-espanholas={len(estranho)}")
    print("   mais comuns fora:", ', '.join(f'{w}({n})' for w, n in c.most_common(12)))
for p in sys.argv[3:]:
    medir(p.rsplit('/',1)[-1], palavras(open(p, encoding='utf-8').read()))
oc = json.load(open(sys.argv[2], encoding='utf-8'))
medir('Tesseract ocr.json', palavras(' '.join(w['t'] for pg in oc.values() for w in pg['words'])))
