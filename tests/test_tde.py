from unittest import TestCase
from torchvision import transforms
import torch
from cdriver.preprocessing.tde import TimeDelayEmbeddingTransform, cropper
import numpy as np
from functools import partial


class TestTimeDelayEmbeddingTransform(TestCase):
    def test_call(self):
        d_embed_x = 2  # incomplete embedding for the variable to be predicted
        d_embed_y = 4  # embedding for the other variable
        tau = 2  # embedding delay
        predict_step_ahead = 0  # number of steps ahead to be predicted
        target_crop = (max(d_embed_x, d_embed_y)-1) * tau + predict_step_ahead  # number of samples to be cropped from the beginning of the time series
        input_crop = (d_embed_y - d_embed_x) * tau

        N = 10
        x = np.arange(N).reshape([-1, 1]).astype(float)
        y = np.arange(N).reshape([-1, 1]).astype(float)
        common_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0), (1)),
                                        torch.Tensor.float,
                                        partial(torch.squeeze, axis=0)])

        y_transform = transforms.Compose([common_transform,
                                          TimeDelayEmbeddingTransform(d_embed_y, tau),
                                          partial(cropper, location='first', n=-input_crop),
                                          partial(cropper, location='last', n=predict_step_ahead) ])

        x_transform = transforms.Compose([common_transform,
                                          TimeDelayEmbeddingTransform(d_embed_x, tau),
                                          partial(cropper, location='first', n=input_crop-1),
                                          partial(cropper, location='last', n=predict_step_ahead+1)])

        target_transform = transforms.Compose([common_transform,
                                               partial(cropper, location='first', n=target_crop)])

        y_transformed = y_transform(y)
        x_transformed = x_transform(x)
        target_transformed = target_transform(x)



        print(y_transformed.shape, x_transformed.shape, target_transformed.shape)
        print(y_transformed)
        print(x_transformed)
        print(target_transformed)
