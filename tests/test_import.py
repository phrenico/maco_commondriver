import unittest

class TestImports(unittest.TestCase):
    def test_import(self):
        import cdriver

    def test_MaCoimport(self):
        from cdriver.network.maco import MaCo

    def test_get_mapper(self):
        from cdriver.network.maco import get_mapper

    def test_get_coach(self):
        from cdriver.network.maco import get_coach

    def test_tent_map(self):
        from cdriver.datagen.tent_map import gen_tentmap