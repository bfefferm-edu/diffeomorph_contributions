### Imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext

from typing import Dict

import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling

from packaging import version

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from typing import Dict, List, Tuple

### Class definitions for SE(3) equivariant layers

class Fiber(object):
    ### This type of data structure allows for specifying different numbers of channels for different degrees / symmetries
    def __init__(self, num_degrees: int=None, num_channels: int=None,
                 structure: List[Tuple[int,int]]=None, dictionary=None):
        if structure:
            self.structure = structure
        elif dictionary:
            self.structure = [(dictionary[o], o) for o in sorted(dictionary.keys())]
        else:
            self.structure = [(num_channels, i) for i in range(num_degrees)]

        self.multiplicities, self.degrees = zip(*self.structure)
        self.max_degree = max(self.degrees)
        self.min_degree = min(self.degrees)
        self.structure_dict = {k: v for v, k in self.structure}
        self.dict = self.structure_dict
        self.n_features = np.sum([i[0] * (2*i[1]+1) for i in self.structure])

        self.feature_indices = {}
        idx = 0
        for (num_channels, d) in self.structure:
            length = num_channels * (2*d + 1)
            self.feature_indices[d] = (idx, idx + length)
            idx += length

    def copy_me(self, multiplicity: int=None):
        s = copy.deepcopy(self.structure)
        if multiplicity is not None:
            # overwrite multiplicities
            s = [(multiplicity, o) for m, o in s]
        return Fiber(structure=s)

    @staticmethod
    def combine(f1, f2):
        new_dict = copy.deepcopy(f1.structure_dict)
        for k, m in f2.structure_dict.items():
            if k in new_dict.keys():
                new_dict[k] += m
            else:
                new_dict[k] = m
        structure = [(new_dict[k], k) for k in sorted(new_dict.keys())]
        return Fiber(structure=structure)

    @staticmethod
    def combine_max(f1, f2):
        new_dict = copy.deepcopy(f1.structure_dict)
        for k, m in f2.structure_dict.items():
            if k in new_dict.keys():
                new_dict[k] = max(m, new_dict[k])
            else:
                new_dict[k] = m
        structure = [(new_dict[k], k) for k in sorted(new_dict.keys())]
        return Fiber(structure=structure)

    @staticmethod
    def combine_selectively(f1, f2):
        # only use orders which occur in fiber f1

        new_dict = copy.deepcopy(f1.structure_dict)
        for k in f1.degrees:
            if k in f2.degrees:
                new_dict[k] += f2.structure_dict[k]
        structure = [(new_dict[k], k) for k in sorted(new_dict.keys())]
        return Fiber(structure=structure)

    @staticmethod
    def combine_fibers(val1, struc1, val2, struc2):
        struc_out = Fiber.combine(struc1, struc2)
        val_out = {}
        for k in struc_out.degrees:
            if k in struc1.degrees:
                if k in struc2.degrees:
                    val_out[k] = torch.cat([val1[k], val2[k]], -2)
                else:
                    val_out[k] = val1[k]
            else:
                val_out[k] = val2[k]
            assert val_out[k].shape[-2] == struc_out.structure_dict[k]
        return val_out

    def __repr__(self):
        return f"{self.structure}"

def get_fiber_dict(F, struc, mask=None, return_struc=False):
    if mask is None: mask = struc
    index = 0
    fiber_dict = {}
    first_dims = F.shape[:-1]
    masked_dict = {}
    for o, m in struc.structure_dict.items():
        length = m * (2*o + 1)
        if o in mask.degrees:
            masked_dict[o] = m
            fiber_dict[o] = F[...,index:index + length].view(list(first_dims) + [m, 2*o + 1])
        index += length
    assert F.shape[-1] == index
    if return_struc:
        return fiber_dict, Fiber(dictionary=masked_dict)
    return fiber_dict


def get_fiber_tensor(F, struc):
    some_entry = tuple(F.values())[0]
    first_dims = some_entry.shape[:-2]
    res = some_entry.new_empty([*first_dims, struc.n_features])
    index = 0
    for o, m in struc.structure_dict.items():
        length = m * (2*o + 1)
        res[..., index: index + length] = F[o].view(*first_dims, length)
        index += length
    assert index == res.shape[-1]
    return res


def fiber2tensor(F, structure, squeeze=False):
    if squeeze:
        fibers = [F[f'{i}'].view(*F[f'{i}'].shape[:-2], -1) for i in structure.degrees]
        fibers = torch.cat(fibers, -1)
    else:
        fibers = [F[f'{i}'].view(*F[f'{i}'].shape[:-2], -1, 1) for i in structure.degrees]
        fibers = torch.cat(fibers, -2)
    return fibers


def fiber2head(F, h, structure, squeeze=False):
    if squeeze:
        fibers = [F[f'{i}'].view(*F[f'{i}'].shape[:-2], h, -1) for i in structure.degrees]
        fibers = torch.cat(fibers, -1)
    else:
        fibers = [F[f'{i}'].view(*F[f'{i}'].shape[:-2], h, -1, 1) for i in structure.degrees]
        fibers = torch.cat(fibers, -2)
    return fibers

### Class definitions for SE(3) equivariant layers

class GConvSE3(nn.Module):
    def __init__(self, f_in, f_out, self_interaction: bool=False, edge_dim: int=0, flavor='skip'):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction
        self.flavor = flavor

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim=edge_dim)

        self.kernel_self = nn.ParameterDict()
        if self_interaction:
            assert self.flavor in ['TFN', 'skip']
            if self.flavor == 'TFN':
                for m_out, d_out in self.f_out.structure:
                    W = nn.Parameter(torch.randn(1, m_out, m_out) / np.sqrt(m_out))
                    self.kernel_self[f'{d_out}'] = W
            elif self.flavor == 'skip':
                for m_in, d_in in self.f_in.structure:
                    if d_in in self.f_out.degrees:
                        m_out = self.f_out.structure_dict[d_in]
                        W = nn.Parameter(torch.randn(1, m_out, m_in) / np.sqrt(m_in))
                        self.kernel_self[f'{d_in}'] = W



    def __repr__(self):
        return f'GConvSE3(structure={self.f_out}, self_interaction={self.self_interaction})'


    def udf_u_mul_e(self, d_out):

        def fnc(edges):
            # Neighbor -> center messages
            msg = 0
            for m_in, d_in in self.f_in.structure:
                src = edges.src[f'{d_in}'].view(-1, m_in*(2*d_in+1), 1)
                edge = edges.data[f'({d_in},{d_out})']
                msg = msg + torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2*d_out+1)

            # Center -> center messages
            if self.self_interaction:
                if f'{d_out}' in self.kernel_self.keys():
                    if self.flavor == 'TFN':
                        W = self.kernel_self[f'{d_out}']
                        msg = torch.matmul(W, msg)
                    if self.flavor == 'skip':
                        dst = edges.dst[f'{d_out}']
                        W = self.kernel_self[f'{d_out}']
                        msg = msg + torch.matmul(W, dst)

            return {'msg': msg.view(msg.shape[0], -1, 2*d_out+1)}
        return fnc

    def forward(self, h, G=None, r=None, basis=None, **kwargs):

        with G.local_scope():
            # Add node features to local graph scope
            for k, v in h.items():
                G.ndata[k] = v

            # Add edge features
            if 'w' in G.edata.keys():
                w = G.edata['w']
                feat = torch.cat([w, r], -1)
            else:
                feat = torch.cat([r, ], -1)

            for (mi, di) in self.f_in.structure:
                for (mo, do) in self.f_out.structure:
                    etype = f'({di},{do})'
                    G.edata[etype] = self.kernel_unary[etype](feat, basis)

            # Perform message-passing for each output feature type
            for d in self.f_out.degrees:
                G.update_all(self.udf_u_mul_e(d), fn.mean('msg', f'out{d}'))

            return {f'{d}': G.ndata[f'out{d}'] for d in self.f_out.degrees}

class GConvSE3Partial(nn.Module):

    def __init__(self, f_in, f_out, edge_dim: int=0, x_ij=None):

        super().__init__()
        self.f_out = f_out
        self.edge_dim = edge_dim

        # adding/concatinating relative position to feature vectors
        # 'cat' concatenates relative position & existing feature vector
        # 'add' adds it, but only if multiplicity > 1
        assert x_ij in [None, 'cat', 'add']
        self.x_ij = x_ij
        if x_ij == 'cat':
            self.f_in = Fiber.combine(f_in, Fiber(structure=[(1,1)]))
        else:
            self.f_in = f_in

        # Node -> edge weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim=edge_dim)

    def __repr__(self):
        return f'GConvSE3Partial(structure={self.f_out})'

    def udf_u_mul_e(self, d_out):

        def fnc(edges):
            # Neighbor -> center messages
            msg = 0
            for m_in, d_in in self.f_in.structure:
                # if type 1 and flag set, add relative position as feature
                if self.x_ij == 'cat' and d_in == 1:
                    # relative positions
                    rel = (edges.dst['x'] - edges.src['x']).view(-1, 3, 1)
                    m_ori = m_in - 1
                    if m_ori == 0:
                        # no type 1 input feature, just use relative position
                        src = rel
                    else:
                        # features of src node, shape [edges, m_in*(2l+1), 1]
                        src = edges.src[f'{d_in}'].view(-1, m_ori*(2*d_in+1), 1)
                        # add to feature vector
                        src = torch.cat([src, rel], dim=1)
                elif self.x_ij == 'add' and d_in == 1 and m_in > 1:
                    src = edges.src[f'{d_in}'].view(-1, m_in*(2*d_in+1), 1)
                    rel = (edges.dst['x'] - edges.src['x']).view(-1, 3, 1)
                    src[..., :3, :1] = src[..., :3, :1] + rel
                else:
                    src = edges.src[f'{d_in}'].view(-1, m_in*(2*d_in+1), 1)
                edge = edges.data[f'({d_in},{d_out})']
                msg = msg + torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2*d_out+1)

            return {f'out{d_out}': msg.view(msg.shape[0], -1, 2*d_out+1)}
        return fnc


    def forward(self, h, G=None, r=None, basis=None, **kwargs):

        with G.local_scope():
            # Add node features to local graph scope
            for k, v in h.items():
                G.ndata[k] = v

            # Add edge features
            if 'w' in G.edata.keys():
                w = G.edata['w'] # shape: [#edges_in_batch, #bond_types]
                feat = torch.cat([w, r], -1)
            else:
                feat = torch.cat([r, ], -1)
            for (mi, di) in self.f_in.structure:
                for (mo, do) in self.f_out.structure:
                    etype = f'({di},{do})'
                    G.edata[etype] = self.kernel_unary[etype](feat, basis)

            # Perform message-passing for each output feature type
            for d in self.f_out.degrees:
                G.apply_edges(self.udf_u_mul_e(d))

            return {f'{d}': G.edata[f'out{d}'] for d in self.f_out.degrees}

class G1x1SE3(nn.Module):

    def __init__(self, f_in, f_out, learnable=True):

        super().__init__()
        self.f_in = f_in
        self.f_out = f_out

        # Linear mappings: 1 per output feature type
        self.transform = nn.ParameterDict()
        for m_out, d_out in self.f_out.structure:
            m_in = self.f_in.structure_dict[d_out]
            self.transform[str(d_out)] = nn.Parameter(torch.randn(m_out, m_in) / np.sqrt(m_in), requires_grad=learnable)

    def __repr__(self):
         return f"G1x1SE3(structure={self.f_out})"

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            if str(k) in self.transform.keys():
                output[k] = torch.matmul(self.transform[str(k)], v)
        return output

class GNormSE3(nn.Module):

    def __init__(self, fiber, nonlin=nn.ReLU(inplace=True), num_layers: int=0):

        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.num_layers = num_layers

        # Regularization for computing phase: gradients explode otherwise
        self.eps = 1e-12

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for m, d in self.fiber.structure:
            self.transform[str(d)] = self._build_net(int(m))

    def __repr__(self):
         return f"GNormSE3(num_layers={self.num_layers}, nonlin={self.nonlin})"

    def _build_net(self, m: int):
        net = []
        for i in range(self.num_layers):
            net.append(BN(int(m)))
            net.append(self.nonlin)
            # TODO: implement cleaner init
            net.append(nn.Linear(m, m, bias=(i==self.num_layers-1)))
            nn.init.kaiming_uniform_(net[-1].weight)
        if self.num_layers == 0:
            net.append(BN(int(m)))
            net.append(self.nonlin)
        return nn.Sequential(*net)


    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # Compute the norms and normalized features
            # v shape: [...,m , 2*k+1]
            norm = v.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(v)
            phase = v / norm

            # Transform on norms
            transformed = self.transform[str(k)](norm[...,0]).unsqueeze(-1)

            # Nonlinearity on norm
            output[k] = (transformed * phase).view(*v.shape)

        return output

### Multi-headed attention, with SE(3) equivariance
class GMABSE3(nn.Module):

    def __init__(self, f_value: Fiber, f_key: Fiber, n_heads: int):

        super().__init__()
        self.f_value = f_value
        self.f_key = f_key
        self.n_heads = n_heads
        self.new_dgl = version.parse(dgl.__version__) > version.parse('0.4.4')

    def __repr__(self):
        return f'GMABSE3(n_heads={self.n_heads}, structure={self.f_value})'

    def udf_u_mul_e(self, d_out):

        def fnc(edges):
            # Neighbor -> center messages
            attn = edges.data['a']
            value = edges.data[f'v{d_out}']

            # Apply attention weights
            msg = attn.unsqueeze(-1).unsqueeze(-1) * value

            return {'m': msg}
        return fnc

    def forward(self, v, k: Dict=None, q: Dict=None, G=None, **kwargs):

        with G.local_scope():
            # Add node features to local graph scope
            ## We use the stacked tensor representation for attention
            for m, d in self.f_value.structure:
                G.edata[f'v{d}'] = v[f'{d}'].view(-1, self.n_heads, m//self.n_heads, 2*d+1)
            G.edata['k'] = fiber2head(k, self.n_heads, self.f_key, squeeze=True) # [edges, heads, channels](?)
            G.ndata['q'] = fiber2head(q, self.n_heads, self.f_key, squeeze=True) # [nodes, heads, channels](?)

            # Compute attention weights
            ## Inner product between (key) neighborhood and (query) center
            G.apply_edges(fn.e_dot_v('k', 'q', 'e'))

            ## Apply softmax
            e = G.edata.pop('e')
            if self.new_dgl:
                # in dgl 5.3, e has an extra dimension compared to dgl 4.3
                # the following, we get rid of this be reshaping
                n_edges = G.edata['k'].shape[0]
                e = e.view([n_edges, self.n_heads])
            e = e / np.sqrt(self.f_key.n_features)
            G.edata['a'] = edge_softmax(G, e)

            # Perform attention-weighted message-passing
            for d in self.f_value.degrees:
                G.update_all(self.udf_u_mul_e(d), fn.sum('m', f'out{d}'))

            output = {}
            for m, d in self.f_value.structure:
                output[f'{d}'] = G.ndata[f'out{d}'].view(-1, m, 2*d+1)

            return output

## Reference: FabianFuchsML, se3-transformer-public/equivariant_attention