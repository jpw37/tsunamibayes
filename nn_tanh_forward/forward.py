import tsunamibayes as tb
from tsunamibayes.forward import BaseForwardModel
import torch
from itertools import chain
import numpy as np
import pandas as pd


class VanillaNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scalers=None):
        super(VanillaNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if scalers is None:
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
                1.0671637 , 39.82251636,  4.39834804,  0.99539551,  1.37684262,
                2.83503784
            ])
            self.out_std = torch.Tensor([
                0.54293611, 0.33864151, 3.57840622, 1.48489459, 0.31817987,
                0.20210712, 2.12708213, 0.6290501 , 0.15368718, 0.21272585,
                0.61394896
            ])
        else:
            self.inp_mean, self.inp_std, self.out_mean, self.out_std = scalers

        self.normalize_input = lambda x: (x - self.inp_mean)/self.inp_std
        self.unnormalize_output = lambda x: x*self.out_std + self.out_mean

        self.input_layer = torch.nn.Linear(
            self.input_size,
            self.hidden_size[0]
        )
        self.input_act = torch.nn.Tanh()
        self.hidden_layers = torch.nn.Sequential(
            *chain.from_iterable(
                [torch.nn.Linear(a,b), torch.nn.Tanh()] for a,b in
                    zip(hidden_size[:-1], hidden_size[1:])
            )
        )
        self.output_layer = torch.nn.Linear(
            self.hidden_size[-1],
            self.output_size
        )

    def forward(self, x):
        with torch.no_grad():
            normalized_x = self.normalize_input(x)
        y = self.input_act(self.input_layer(normalized_x))
        z = self.hidden_layers(y)
        return self.output_layer(z)

    def forward_unnormalized(self, x):
        with torch.no_grad():
            return self.unnormalize_output(self.forward(x))


class NeuralNetEmulator(BaseForwardModel):
    obstypes = ['arrival','height','inundation']
    # NOTE: the ordering on input/output columns matters. Do not change it.
    nn_input_cols = ['latitude', 'longitude', 'length', 'width', 'slip',
        'strike', 'dip', 'depth'
    ]
    nn_output_cols = ['Pulu Ai height', 'Ambon height', 'Banda Neira arrival',
        'Banda Neira height', 'Buru height', 'Hulaliu height',
        'Saparua arrival', 'Saparua height', 'Kulur height', 'Ameth height',
        'Amahai height'
    ]
    def __init__(
        self,
        gauges,
        fault,
        nn_weight_path='./data/weights.pt',
        input_size=8,
        hidden_size=[100]*13,
        output_size=11,
    ):
        """
        Parameters
        ----------
        gauges : (list) of Gauge objects
            The list of gauge objects containing location data and distribution
            functions for the arrival, height, and inundation parameters for
            each gauge location.
        fault : Fault object
            Usually a previously constructed GridFault object.
        """
        super().__init__(gauges)
        self.fault = fault
        self.net = VanillaNet(
            input_size=input_size,
            hidden_size=[100]*13,
            output_size=output_size,
        )
        self.net.load_state_dict(torch.load(nn_weight_path))
        self.net.eval()

    def run(self,model_params,verbose=False):
        """Runs the forward model for a specified sample's model parameters,
        then returns the computed arrival times, wave heights, and inundation distances
        for each gauge location, as appropriate.
        Essentially, this function works 'backwards' from a sample model of the fault rupture's
        parameters to compute what the gauge's observations would have been like for that given
        earthquake event sample.

        Parameters
        ----------
        model_params : dict
            The sample's model parameters. The dictionary whose keys are the okada parameters:
            'latitude', 'longitude', 'depth_offset', 'strike','length','width','slip','depth',
            'dip','rake', and whose associated values are floats.
        verbose : bool
            Flag for verbose output, optional. Default is False.

        Returns
        -------
        model_output : pandas Series
            A pandas series whose axes labels are the combinations of the scenario's gauges
            names plus 'arrivals', 'height', or 'inundation'. The associated values are floats.
        """
        model_params['depth'] += model_params['depth_offset']


        X = torch.Tensor([model_params[x] for x in self.nn_input_cols])
        y = np.array(self.net.forward_unnormalized(X))

        model_output = pd.Series(dtype='float64')
        for idx, output_col in enumerate(self.nn_output_cols):
            model_output[output_col] = y[idx]

        return model_output

    def llh(self,model_output,verbose=False):
        """Comptues the loglikelihood of the forward model's ouput.
        Compares the model outputs with the acutal gauge observation distributions for
        arrival time, wave height, and inundation distance, and calculates the log of
        the probability distribution function at each point of the model's output for the gagues.
        Finally, sums together all the log values for each observation point and type.

        Parameters
        ----------
        model_output : pandas Series
            A pandas series whose axes labels are the cominations of the scenario's gauges
            names plus 'arrivals', 'height', or 'inundation'. The associated values are floats.
        verbose : bool
            Flag for verbose output, optional. Default is False.
            If set to true, will output a Gauge Log with each gauges computed arrival time,
            wave height, and inundation distance with the associted loglikelihood.

        Returns
        -------
        llh : float
            The loglikelihood of the given sample's model output. A measure of how likely
            that model output is when compared to the actual observation data at each gauge.
        """
        llh = 0
        if verbose: print("Gauge Log\n---------")
        for gauge in self.gauges:
            if verbose: print(gauge.name)
            if 'arrival' in gauge.obstypes:
                arrival_time = model_output[gauge.name+' arrival']
                log_p = gauge.dists['arrival'].logpdf(arrival_time)
                llh += log_p
                if verbose: print("arrival:    {:.3f}\tllh: {:.3e}".format(arrival_time,log_p))

            if 'height' in gauge.obstypes:
                wave_height = model_output[gauge.name+' height']
                if np.abs(wave_height) > 999999999: log_p = np.NINF
                else: log_p = gauge.dists['height'].logpdf(wave_height)
                llh += log_p
                if verbose: print("height:     {:.3f}\tllh: {:.3e}".format(wave_height,log_p))

            if 'inundation' in gauge.obstypes:
                inundation = model_output[gauge.name+' inundation']
                log_p = gauge.dists['inundation'].logpdf(inundation)
                llh += log_p
                if verbose: print("inundation: {:.3f}\tllh: {:.3e}".format(inundation,log_p))
        return llh

    def write_fgmax_grid(self,gauges,fgmax_params):
        """Writes a file to store a specific scenario's parameters for the
        fixed grid maximum monitoring feature in GeoClaw.
        After it is written, the file at the specific path will contain values for each
        fixed grid parameter, the total number of observations from the gauges, and the
        gauge location information.

        Parameters
        ----------
        gauges : (list) of Gauge objects
            The list of gauge objects containing locations data distribution functions for the
            arrival, height, and inundation for each gauge location.
        fgmax_params : dict
            The dictionary containing information for the fixed grid maximums (see GeoClaw pack)
            with keys 'min_level_check', 'fgmax_grid_path', 'valuemax_path',
            'aux1_path', as well as others specified in the scenario's defaults.cfg file.
            Associated values are a variety of strings for file paths & floats.
        """
        npts = sum(1 for gauge in gauges if
                   any(obstype in self.obstypes for obstype in gauge.obstypes))

        with open(fgmax_params['fgmax_grid_path'],'w') as f:
            f.write(str(fgmax_params['tstart_max'])+'\t# tstart_max\n')
            f.write(str(fgmax_params['tend_max'])+'\t# tend_max\n')
            f.write(str(fgmax_params['dt_check'])+'\t# dt_check\n')
            f.write(str(fgmax_params['min_level_check'])+'\t# min_level_check\n')
            f.write(str(fgmax_params['arrival_tol'])+'\t# arrival_tol\n')
            f.write('0'+'\t# point_style\n')
            f.write(str(npts)+'\t# n_pts\n')
            for gauge in gauges:
                if any(obstype in self.obstypes for obstype in gauge.obstypes):
                    f.write(str(gauge.lon)+' '+str(gauge.lat))
                    f.write('\t# '+gauge.name+'\n')
