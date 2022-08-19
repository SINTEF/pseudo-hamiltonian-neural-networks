from casadi import *
import casadi.tools


class CasadiFCNN:
    """
    Casadi-implementation of a fully-connected neural network. The class
    takes in a specification of a neural network
    model and implements the specification as a Casadi function.

    parameters
    ----------
        layers : list of dicts
            List of dictionaries where each entry describes a layer of
            the neural network as returned by the
            "get_pytorch_model_architecture" function.
    """
    def __init__(self, layers=()):
        self.layers = layers
        self.params = None
        self.params_num = None

        self.forward_fn = None

    def create_forward(self, *args):
        input_data_cat = vertcat(*args).T
        blank = SX.sym('blank')  # placeholder for activation function input

        # neuron_weights=SX.sym('neuron_weights')
        in_size = input_data_cat.shape[1]
        h_l_ps = []

        act_funs = {'tanh': Function('tanh_f', [blank], [casadi.tanh(blank)]),
                    'relu': Function('relu_f', [blank], [casadi.fmax(0, blank)])}

        hidden_layer = input_data_cat
        for l_i, layer in enumerate(self.layers):
            if layer['type'] == 'activation':
                assert layer['name'] in act_funs
                hidden_layer = act_funs[layer['name']](hidden_layer)
            elif layer['type'] == 'linear':
                l_ws = casadi.tools.entry(f'hl_{l_i}_weights', sym=SX.sym(f'hl_{l_i}_weights', hidden_layer.shape[1], layer['out_features']))
                h_l_ps.append(l_ws)

                hidden_layer = (hidden_layer @ l_ws.sym)

                if layer['bias']:
                    l_bs = casadi.tools.entry(f'hl_{l_i}_bias', sym=SX.sym(f'hl_{l_i}_bias', layer['out_features'], 1))
                    h_l_ps.append(l_bs)
                    hidden_layer = hidden_layer + (DM.ones(hidden_layer.shape[0], layer['out_features']) @ diag(l_bs.sym))
            else:
                raise ValueError(f'Unsupported layer type {layer["type"]}, must be one of [linear, activation]')

        self.params = casadi.tools.struct_symSX(h_l_ps)
        self.params_num = np.zeros(self.params.shape)

        self.forward_fn = Function('nn_forward', list(args) + [self.params], [hidden_layer])

        return hidden_layer

    def set_weights_and_biases(self, params, source='pytorch'):
        if source == 'tensorflow':
            self.params_num = np.concatenate([p.flatten(order='F') for p in params]).reshape(-1, 1)
        elif source == 'pytorch':
            self.params_num = np.concatenate([p.flatten() for p in params]).reshape(-1, 1)
        else:
            raise ValueError

    def get_parameters(self):
        return self.params_num

    def test_correct_output(self, gt_model, n=10):
        for i in range(n):
            x = np.random.normal(0, 1, size=self.params.entries[0].sym.shape[0])
            assert np.isclose(self.forward_fn(x, self.params_num), gt_model(x)).all()


def get_pytorch_model_parameters(model):
    res = []
    for p_name, p in model.named_parameters():
        if 'weight' in p_name:
            res.append(p.detach().numpy())
        elif 'bias' in p_name:
            res.append(p.detach().numpy())
        else:
            raise ValueError

    return res


def get_pytorch_model_architecture(model):
    res = []
    for l_i, layer in enumerate(model.modules()):
        if l_i <= 1:
            continue
        l = {'name': str(layer).split('(')[0].lower()}
        if not hasattr(layer, 'out_features'):  # is an activation layer
            l['type'] = 'activation'
            res.append(l)
        else:
            l['type'] = 'linear'
            l['bias'] = layer.bias is not None
            l['out_features'] = layer.out_features
            res.append(l)

    return res
