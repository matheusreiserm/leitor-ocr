"""Testes dos invariantes: py -I -X utf8 testes.py (da pasta do repositório)."""
import ast, sys, unittest
from pathlib import Path
import numpy as np, cv2

AQ = Path(__file__).resolve().parent; sys.path.insert(0, str(AQ))


def _wer():
    """A função wer do medir.py, sem executar o resto do módulo (que roda OCR ao importar)."""
    arv = ast.parse((AQ / 'medir.py').read_text(encoding='utf-8'))
    f = next(n for n in arv.body if isinstance(n, ast.FunctionDef) and n.name == 'wer')
    ns = {}; exec(compile(ast.Module(body=[f], type_ignores=[]), 'wer', 'exec'), ns)
    return ns['wer']


class WER(unittest.TestCase):
    def test_minimo(self):
        w = _wer()
        self.assertEqual(w(['a', 'a'], ['b', 'a']), 1)          # o SequenceMatcher contava 2
        self.assertEqual(w(['a', 'b'], ['a', 'c', 'b']), 1)
        self.assertEqual(w(['a', 'b'], ['a']), 1)
        self.assertEqual(w(list('abcd'), list('bcda')), 2)
        self.assertEqual(w(['la', 'vida'], ['la', 'vida']), 0)
        self.assertEqual(w([], ['x', 'y']), 2)

    def test_simetrica(self):
        w = _wer(); a, b = ['de', 'la', 'vida', 'nueva'], ['la', 'vida', 'nueva', 'ahora']
        self.assertEqual(w(a, b), w(b, a))


class Protecao(unittest.TestCase):
    """Página sintética: "mi" com o pingo do i alto, entre 2,1 e 2,3 alturas de x da base: fora da faixa
    da linha (seria apagado "entre linhas") mas dentro da zona de marcas, onde a palavra o pede."""

    def _pagina(self, xh=20, b=200):
        g = np.full((400, 400), 255, np.uint8)
        cv2.rectangle(g, (100, b - xh), (130, b - 1), 0, -1)         # m
        cv2.rectangle(g, (140, b - xh), (145, b - 1), 0, -1)         # haste do i
        y = int(b - 2.3 * xh); cv2.rectangle(g, (140, y), (145, y + 3), 0, -1)   # pingo: topo a 2,3 xh, centro acima de 2,1 xh
        cv2.rectangle(g, (300, 50), (304, 54), 0, -1)                # sujeira solta, longe de tudo
        pg = {'words': [dict(t='mi', conf=96, x=100, y=b - xh - 50, w=46, h=xh + 50, line=[1, 1, 1], wn=1)],
              'chars': [['m', 100, b - xh, 131, b], ['i', 140, b - xh, 146, b]]}
        return g, pg

    def test_pingo_protegido_nao_some(self):
        import limpar_por_moldes as L
        L.ONDE_MARCA.clear(); L.ONDE_MARCA['i'] = (0.0, -2.2 * 20)   # o molde diz: pingo alto, nesta fonte
        g, pg = self._pagina()
        formas = L.Formas.__new__(L.Formas)
        formas.validar = lambda t: t                      # aceita a palavra como está
        out, m, motivo = L.limpar(g, pg, formas)
        self.assertEqual(out[154:158, 140:146].min(), 0, 'o pingo do i foi apagado')
        self.assertEqual(out[50:55, 300:305].min(), 255, 'a sujeira solta deveria sair')
        self.assertEqual(motivo['entre linhas'], 1)


class AcentoVerbal(unittest.TestCase):
    def test_preterito_nao_vira_substantivo(self):
        import limpar_por_moldes as L
        self.assertTrue(L.acento_verbal_possivel('aleteó', 'aleteo'))     # o acento era real (página 25)
        self.assertTrue(L.acento_verbal_possivel('negoció', 'negocio'))
        self.assertTrue(L.acento_verbal_possivel('moduló', 'módulo'))      # -lo do radical, não pronome (página 23)
        self.assertFalse(L.acento_verbal_possivel('parecidó', 'parecido'))  # particípio: sempre sujeira
        self.assertFalse(L.acento_verbal_possivel('preguntandó', 'preguntando'))
        self.assertFalse(L.acento_verbal_possivel('taparló', 'taparlo'))
        self.assertFalse(L.acento_verbal_possivel('oyó', 'oyó'))
        self.assertFalse(L.acento_verbal_possivel('estabá', 'estaba'))


class Portugues(unittest.TestCase):
    def test_marcas_que_mudam_a_palavra_ficam(self):
        import limpar_por_moldes as L
        ant = L.idioma.ATUAL; L.idioma.definir(codigo='por')
        try:
            f = L.Formas(L.T.Lexico(set()))
            for w in ('pôde', 'nós', 'às', 'não', 'você', 'coração'): self.assertEqual(f.validar(w), w)
            for w in ('idéia', 'vôo', 'freqüente', 'pára'): self.assertIsNone(f.validar(w))   # grafia antiga: não mexe
            self.assertEqual(f.validar('câsa'), 'casa')            # marca que não existe na palavra: sujeira
        finally:
            L.idioma.definir(codigo=ant)


if __name__ == '__main__':
    unittest.main(verbosity=2)
