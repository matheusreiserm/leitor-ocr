import unittest
from avaliar_gabarito import avaliar, distancia_edicao

class TestarGabarito(unittest.TestCase):
    def test_wermetrica_substituicao(self):
        self.assertEqual(distancia_edicao(['a','a'], ['b','a'])[0], 1)
    def test_pontuacao_nao_conta(self):
        m, e = avaliar('La vida nueva.', 'LA, vida nueva!')
        self.assertEqual(m['erros_palavra'], 0)
        self.assertEqual(m['erros_caractere'], 0)
    def test_diacritico_conta(self):
        m, e = avaliar('había cabaña', 'habia cabana')
        self.assertEqual(m['erros_palavra'], 2)
        self.assertEqual(m['substituicoes_so_diacriticos'], 2)
    def test_palavra_inserida(self):
        m, e = avaliar('la nueva', 'la vida nueva')
        self.assertEqual(m['erros_palavra'], 1)
    def test_referencia_vazia_falha(self):
        with self.assertRaises(ValueError): avaliar('', 'hola')
if __name__ == '__main__': unittest.main(verbosity=2)
