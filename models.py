import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaSequential)

# from torchmeta.modules.utils import get_subdict


# Code adapted from Sitzmann et al 2020
# https://github.com/vsitzmann/siren

def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


class BatchLinear(nn.Linear, MetaModule):
    """
    A linear meta-layer that can deal with batched weight matrices
    and biases, as for instance output by a hypernetwork.
    """
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class FCBlock(MetaModule):
    """
    A fully connected MLP that also allows swapping out the weights
    when used with a hypernetwork.
    """

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()
        self.first_layer_init = None

        nls_and_inits = {
            'relu':     (nn.ReLU(inplace=True), init_weights_normal, None),
            'sigmoid':  (nn.Sigmoid(), init_weights_xavier, None),
            'tanh':     (nn.Tanh(), init_weights_xavier, None),
            'selu':     (nn.SELU(inplace=True), init_weights_selu, None),
            'softplus': (nn.Softplus(), init_weights_normal, None),
            'elu':      (nn.ELU(inplace=True), init_weights_elu, None)
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None: 
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        # First layer
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))
        # Hidden layers
        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))
        # Output layer
        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=self.get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        """
        Returns not only model output, but also intermediate activations.
        """
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = self.get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=self.get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations[f'{sublayer.__class__}_{i}'] = x
        return activations


class SetEncoder(nn.Module):
    """
    Takes per-sample coordinates & amplitudes (and optionally other features),
    merges them, and outputs a latent embedding for the hypernetwork.
    """

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, nonlinearity='relu'):
        super().__init__()

        assert nonlinearity in ['relu'], 'Currently only ReLU is implemented here'
        if nonlinearity == 'relu':
            nl = nn.ReLU(inplace=True)
            weight_init = init_weights_normal

        self.net = [nn.Linear(in_features, hidden_features), nl]
        for _ in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))
        self.net.append(nn.Linear(hidden_features, out_features))
        # No final activation here? We can keep it as is or add another ReLU

        self.net = nn.Sequential(*self.net)
        self.net.apply(weight_init)

    def forward(self, context_x, context_y, **kwargs):
        """
        context_x: [N, Xdim] (e.g. coords + optional param)
        context_y: [N, Ydim] (e.g. amplitudes)

        We concatenate them => shape [N, Xdim + Ydim],
        pass through MLP => [N, out_features],
        then average over samples => shape [out_features].
        """
        input_ = torch.cat((context_x, context_y), dim=-1)
        embeddings = self.net(input_)              # shape [N, out_features]
        return embeddings.mean(dim=-2)             # shape [out_features]


class HyperNetwork(nn.Module):
    """
    Takes a latent embedding 'z' and generates the weights for a 'hypo_module'.
    """
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()
        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []

        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            out_dim = int(torch.prod(torch.tensor(param.size())))
            hn = FCBlock(in_features=hyper_in_features,
                         out_features=out_dim,
                         num_hidden_layers=hyper_hidden_layers,
                         hidden_features=hyper_hidden_features,
                         outermost_linear=True,
                         nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        """
        z: [latent_dim] or [batch, latent_dim]
        returns an OrderedDict of {parameter_name: parameter_values}
        that can be fed as 'params' to the hypo_module.
        """
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            raw = net(z)  # shape [batch, out_dim]
            reshaped = raw.reshape(batch_param_shape)
            params[name] = reshaped
        return params


class SingleBVPNet(MetaModule):
    """
    The 'hypo' network that produces the final BRDF outputs from 6D coords.
    """

    def __init__(self, out_features=1, type='relu', in_features=6, hidden_features=256, num_hidden_layers=3):
        super().__init__()
        self.mode = 'mlp'
        self.net = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=True,
                           nonlinearity=type)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        output = self.net(coords, self.get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}


def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2
    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1 / in_features_main_net, 1 / in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2
    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1 / fan_in, 1 / fan_in)


class HyperBRDF(nn.Module):
    """
    Overall pipeline:
    - We have a 'set encoder' that sees coords + amps + (optionally) 2 param dims
      to produce a latent code 'embedding'.
    - That embedding is fed to a HyperNetwork, which returns the 'hypo' net's weights.
    - The hypo net (SingleBVPNet) then uses those weights to compute the final BRDF.

    By default, SingleBVPNet is in_features=6 (just hx,hy,hz,dx,dy,dz). The 2 param dims
    are used only inside the set encoder => they shape the net's weights, but are not
    direct input to SingleBVPNet.

    If you want to feed params directly as part of the final net's input, you can change
    SingleBVPNet(in_features=8).
    """

    def __init__(self, in_features=6, out_features=3, encoder_nl='relu'):
        super().__init__()

        self.latent_dim = 40

        # (CHANGED) If your set encoder sees 'coords' of dim=6 plus 'amps' of dim=3 plus 'params' of dim=2 => 11
        self.set_encoder_in_dim = 6 + 3 + 2  # (ADDED) = 11

        self.set_encoder = SetEncoder(
            in_features=self.set_encoder_in_dim,
            out_features=self.latent_dim,
            num_hidden_layers=2,
            hidden_features=128,
            nonlinearity=encoder_nl
        )

        # The "hypo" net expects 6D input (the directions).  If you want 8D, change in_features=8 below.
        self.hypo_net = SingleBVPNet(
            out_features=out_features,
            hidden_features=60,
            type='relu',
            in_features=in_features  # typically 6
        )

        # The hyper net maps from 'latent_dim' => weights of self.hypo_net
        self.hyper_net = HyperNetwork(
            hyper_in_features=self.latent_dim,
            hyper_hidden_layers=1,
            hyper_hidden_features=128,
            hypo_module=self.hypo_net
        )

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def forward(self, model_input):
        """
        model_input is expected to contain:
          - 'coords' -> [N,6]
          - 'amps'   -> [N,3]
          - 'params' -> [2] (thickness, doping), or Nx2 if you prefer

        We'll incorporate 'params' into the set encoder's input by
        broadcasting it across all N samples, then concatenating.
        """
        coords = model_input['coords']   # shape [N,6]
        amps   = model_input['amps']     # shape [N,3]

        
        p = model_input.get("params", None)

        if p is None:
            # default to zeros if no params
            B, N, _ = coords.shape
            p = torch.zeros((B, N, 2), device=coords.device)

        else:
            # Check how many dims 'p' has
            if p.dim() == 1:
                # shape [2], expand to [B,N,2]
                B, N, _ = coords.shape
                p = p.unsqueeze(0).unsqueeze(1).expand(B, N, 2)  
            elif p.dim() == 2:
                # shape [B,2] => expand to [B,N,2]
                B, N, _ = coords.shape
                p = p.unsqueeze(1).expand(B, N, 2)
            elif p.dim() == 3:
                # shape [B,N,2], already good
                pass
            else:
                raise ValueError(f"Unsupported 'params' shape: {p.shape}")

        # Now coords is [B,N,6], p is [B,N,2], so we can safely cat:
        coords_plus_param = torch.cat([coords, p], dim=-1)  # shape [B,N,8]

        # Next, cat with amps => shape [N, (8 + 3)=11]
        # The set encoder is dimension 11
        embedding = self.set_encoder(coords_plus_param, amps)  # => shape [latent_dim]

        # hyper_net => get the parameters for the hypo_net
        hypo_params = self.hyper_net(embedding)  # dict of weight/bias for SingleBVPNet

        # Finally, run SingleBVPNet with those newly generated weights
        out_dict = self.hypo_net({'coords': coords}, params=hypo_params)

        return {
            'model_in': out_dict['model_in'],   # [N,6]
            'model_out': out_dict['model_out'], # [N, out_features]
            'latent_vec': embedding,            # [latent_dim]
            'hypo_params': hypo_params
        }
