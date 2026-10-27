"""Idioma do livro, lido de <pasta>/reconstrucao/config.json -> {"idioma": "por"}. Sem a chave, espanhol.

Cada script chama definir(pasta) no começo; os demais leem ATUAL. O português tem duas diferenças que
importam para a limpeza e a revisão:
- marcas próprias: til em ã/õ (traço horizontal sobre a ou o, que no espanhol só existe no ñ) e circunflexo;
- livros anteriores ao Acordo Ortográfico (em vigor desde 2009) usam acentos que o léxico, feito de legendas
  recentes, não traz: "idéia", "jóia", "vôo", "vêem", "freqüente", "pára", "pêlo". moderna() leva a grafia
  antiga à atual, para que o léxico a reconheça em vez de tomar o acento por sujeira.
"""
import json, re
from pathlib import Path

IDIOMAS = {'spa': dict(tess='spa', lexico='es.json.gz'), 'por': dict(tess='por', lexico='pt.json.gz')}
ATUAL = 'spa'
GRAFIA_ANTIGA = False   # livro em português anterior a 2009: trema nas famílias de palavras que o tinham


def do_livro(pasta):
    f = Path(pasta) / 'reconstrucao' / 'config.json'
    cfg = json.load(open(f, encoding='utf-8-sig')) if f.exists() else {}
    return cfg.get('idioma', 'spa')


def definir(pasta=None, codigo=None):
    global ATUAL, GRAFIA_ANTIGA
    ATUAL = codigo or do_livro(pasta)
    if pasta is not None:
        f = Path(pasta) / 'reconstrucao' / 'config.json'
        GRAFIA_ANTIGA = bool((json.load(open(f, encoding='utf-8-sig')) if f.exists() else {}).get('grafia_antiga'))
    if ATUAL not in IDIOMAS: raise ValueError(f'idioma desconhecido: {ATUAL} (conhecidos: {", ".join(IDIOMAS)})')
    return ATUAL


def tesseract(): return IDIOMAS[ATUAL]['tess']
def lexico(): return IDIOMAS[ATUAL]['lexico']


# grafia anterior ao Acordo de 1990 (vigente no Brasil desde 2009) -> grafia atual
_DIFERENCIAIS = {'pára': 'para', 'pêlo': 'pelo', 'pêlos': 'pelos', 'pélo': 'pelo', 'pólo': 'polo', 'pólos': 'polos',
                 'pêra': 'pera', 'pêras': 'peras', 'côa': 'coa', 'pôlo': 'polo'}


def moderna(w):
    """'idéia' -> 'ideia', 'vôo' -> 'voo', 'freqüente' -> 'frequente', 'pára' -> 'para' (só português)."""
    if ATUAL != 'por': return w
    low = w.lower()
    if low in _DIFERENCIAIS: return _DIFERENCIAIS[low]
    s = low.replace('ü', 'u')
    s = re.sub(r'([éó])i(?=[aeo]s?$|[aeo]$)', lambda m: {'é': 'e', 'ó': 'o'}[m.group(1)] + 'i', s)   # idéia, jóia, platéia
    s = re.sub(r'ói(?=c)', 'oi', s)                                                                  # heróico, paranóico
    s = re.sub(r'ôo(?=s?$)', 'oo', s)                                                                # vôo, enjôo, perdôo
    s = re.sub(r'êem$', 'eem', s)                                                                    # vêem, lêem, crêem
    return s


def grafia_antiga(w):
    """A palavra tem uma marca que só a grafia antiga usa (o léxico atual não a conhece)."""
    return ATUAL == 'por' and moderna(w) != w.lower()


# famílias que tinham trema antes de 2009 (o u de gue/gui/que/qui se pronunciava): sempre, sem exceção dentro da família
_TREMA = re.compile(r'(?<![a-záéíóúâêôãõç])(ag[uü](?=ent)|ling[uü](?=[ií])|bil[ií]ng[uü](?=e)|freq[uü](?=[eê]n)|conseq[uü](?=[eê]n)|'
                    r'seq[uü](?=e[lsn])|tranq[uü](?=il)|cinq[uü](?=ent)|ping[uü](?=i[mn])|sang[uü](?=[ií]n)|'
                    r'eq[uü](?=i(?:ta|d|n|l))|delinq[uü](?=[eê]n)|eloq[uü](?=[eê]n)|ung[uü](?=ent)|arg[uü](?=i)|'
                    r'ag[uü](?=e)|averig[uü](?=e)|apazig[uü](?=e)|enxag[uü](?=e)|sag[uü](?=i)|q[uü](?=inq[uü])|'
                    r'quinq[uü](?=[eê]n)|ubiq[uü](?=i)|eq[uü](?=estre))', re.I)


def com_trema(w):
    """'aguenta' -> 'agüenta', 'linguística' -> 'lingüística', em livro de grafia antiga (senão, a palavra como está)."""
    if ATUAL != 'por' or not GRAFIA_ANTIGA: return w
    return _TREMA.sub(lambda m: m.group(0)[:-1] + ('Ü' if m.group(0)[-1].isupper() else 'ü'), w)
