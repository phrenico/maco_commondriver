import unittest

import torch

from cdriver.preprocessing.tde import TimeDelayEmbeddingTransform, cropper


class TestTimeDelayEmbeddingTransform(unittest.TestCase):
    def test_incomplete_embedding_alignment(self):
        x = torch.arange(10, dtype=torch.float32).view(-1, 1)
        y = torch.arange(10, dtype=torch.float32).view(-1, 1)

        y_transformed = cropper(TimeDelayEmbeddingTransform(4, 2)(y), n=-4, location='first')
        y_transformed = cropper(y_transformed, n=0, location='last')

        x_transformed = cropper(TimeDelayEmbeddingTransform(2, 2)(x), n=3, location='first')
        x_transformed = cropper(x_transformed, n=1, location='last')

        target_transformed = cropper(x.squeeze(-1), n=6, location='first')

        expected_y = torch.tensor(
            [[0.0, 2.0, 4.0, 6.0], [1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0], [3.0, 5.0, 7.0, 9.0]]
        )
        expected_x = torch.tensor([[3.0, 5.0], [4.0, 6.0], [5.0, 7.0], [6.0, 8.0]])
        expected_target = torch.tensor([6.0, 7.0, 8.0, 9.0])

        torch.testing.assert_close(y_transformed, expected_y)
        torch.testing.assert_close(x_transformed, expected_x)
        torch.testing.assert_close(target_transformed, expected_target)

    def test_predict_step_target_split(self):
        x = torch.arange(5, dtype=torch.float32).view(-1, 1)
        y = torch.arange(5, dtype=torch.float32).view(-1, 1)

        y_transformed = cropper(TimeDelayEmbeddingTransform(2, 1)(y), n=0, location='first')
        transformed = cropper(TimeDelayEmbeddingTransform(2, 1)(x), n=0, location='first')
        x_transformed = transformed[:, :-1]
        target_transformed = transformed[:, -1:]

        expected_y = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        expected_x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        expected_target = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

        torch.testing.assert_close(y_transformed, expected_y)
        torch.testing.assert_close(x_transformed, expected_x)
        torch.testing.assert_close(target_transformed, expected_target)

    def test_cropper_rejects_invalid_location(self):
        with self.assertRaises(ValueError):
            cropper(torch.arange(5, dtype=torch.float32), n=2, location='middle')