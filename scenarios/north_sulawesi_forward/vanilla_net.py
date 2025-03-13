import torch
from itertools import chain


class VanillaNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scalers=None):
        super(VanillaNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if scalers is None:
            # Here I've hardcoded the mean_'s and scale_'s of input/output scalers.
            self.inp_mean = torch.Tensor([
                -4.59936018e+00,  1.31706962e+02,  3.80556213e+05,  1.05726545e+05,
                1.22070403e+01,  1.49188727e+02,  1.48334349e+01,  1.73222231e+04
            ])
            self.inp_std = torch.Tensor([
                5.35557343e-01, 2.00745698e-01, 1.22149250e+05, 2.86161126e+04,
                3.68246696e+00, 2.23304735e+01, 3.15135316e+00, 5.14069803e+03
            ])

            self.out_mean = torch.Tensor([
                1.70409856,  2.36874371, 14.85654644,  5.24019232,  1.51261909,
                1.0671637, 39.82251636,  4.39834804,  0.99539551,  1.37684262,
                2.83503784
            ])
            self.out_std = torch.Tensor([
                0.54293611, 0.33864151, 3.57840622, 1.48489459, 0.31817987,
                0.20210712, 2.12708213, 0.6290501, 0.15368718, 0.21272585,
                0.61394896
            ])
        else:
            inp_scaling, out_scaling = scalers
            self.inp_mean, self.inp_std = inp_scaling
            self.out_mean, self.out_std = out_scaling

        self.normalize_input = lambda x: (x - self.inp_mean) / self.inp_std
        self.unnormalize_output = lambda x: x * self.out_std + self.out_mean

        self.input_layer = torch.nn.Linear(
            self.input_size, self.hidden_size[0])
        self.input_act = torch.nn.ReLU()
        self.hidden_layers = torch.nn.Sequential(
            *chain.from_iterable(
                [torch.nn.Linear(a, b), torch.nn.ReLU()] for a, b in zip(hidden_size[:-1], hidden_size[1:])
            )
        )
        self.output_layer = torch.nn.Linear(
            self.hidden_size[-1], self.output_size)

    def forward(self, x):
        normalized_x = self.normalize_input(x)
        y = self.input_act(self.input_layer(normalized_x))
        z = self.hidden_layers(y)
        return self.output_layer(z)

    def forward_unnormalized(self, x):
        return self.unnormalize_output(self.forward(x))
