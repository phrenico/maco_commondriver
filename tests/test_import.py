import unittest


class TestImports(unittest.TestCase):
    def test_import_package(self):
        import cdriver

        self.assertIsNotNone(cdriver)

    def test_import_maco_symbols(self):
        from cdriver.network.maco import MaCo, get_coach, get_mapper

        self.assertIsNotNone(MaCo)
        self.assertTrue(callable(get_coach))
        self.assertTrue(callable(get_mapper))

    def test_import_tent_map_generator(self):
        from cdriver.datagen.tent_map import gen_tentmapdata

        self.assertTrue(callable(gen_tentmapdata))