import e3nn_jax as e3nn
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx

from .aggregation import AttentionAggregation #, AttentionAggregationV2
from .aggregation import AttentionAggregation_SE3_transformer


class NeuralVectorField(eqx.Module):
    aggr: AttentionAggregation
    force: eqx.nn.MLP
    # force: Optional[eqx.nn.MLP]
    # force_short: Optional[eqx.nn.MLP]
    # force_long: Optional[eqx.nn.MLP]
    num_genes: int
    time_dependent: bool
    distinguish_range: bool
    use_rel_dist_pe: bool
    use_rel_angle_pe: bool

    def __init__(self, 
        num_genes,
        num_morphogens, # size for QKV
        num_heads,
        hidden_size,
        num_layers,
        radius,
        time_dependent,
        distinguish_range,
        use_rel_dist_pe,
        use_rel_angle_pe,
        key,
    ):
        self.num_genes = num_genes
        self.time_dependent = time_dependent
        self.distinguish_range = distinguish_range
        self.use_rel_dist_pe = use_rel_dist_pe
        self.use_rel_angle_pe = use_rel_angle_pe
        
        key_attn, key_force = jrandom.split(key, 2)
        if distinguish_range:
            raise NotImplementedError(
                """We haven't made AttentionAggregation to distinguish
                short-ranged and long-ranged interaction.""")
        else:
            self.aggr = AttentionAggregation(
                in_features=num_genes,
                out_features=num_morphogens,
                num_heads=num_heads,
                time_dependent=time_dependent,
                use_rel_dist_pe=use_rel_dist_pe,
                use_rel_angle_pe=use_rel_angle_pe,
                key=key_attn,
            )
        force_in_size = num_genes + num_morphogens
        if time_dependent:
            force_in_size += 1
        if distinguish_range:
            raise NotImplementedError(
                """We may compute two separate force models for
                attention-aggregated morphogens from short/long-ranged neighbors.""")
            # key_force_short, key_force_long = jrandom.split(key_force, 2)
            # self.force_short = eqx.nn.MLP(
            #     in_size=force_in_size,
            #     out_size=3+num_genes+3,
            #     width_size=hidden_size,
            #     depth=num_layers,
            #     activation=jax.nn.swish,
            #     #activation=jax.nn.softplus,
            #     #activation=jax.nn.gelu,
            #     #final_activation=lambda x: x,
            #     final_activation=lambda x: x,
            #     key=key_force_short,
            # )   
            # self.force_long = eqx.nn.MLP(
            #     in_size=force_in_size,
            #     out_size=3+num_genes+3,
            #     width_size=hidden_size,
            #     depth=num_layers,
            #     activation=jax.nn.swish,
            #     #activation=jax.nn.softplus,
            #     #activation=jax.nn.gelu,
            #     #final_activation=lambda x: x,
            #     final_activation=lambda x: x,
            #     key=key_force_long,
            # )    
            # self.force = None
        else:
            self.force = eqx.nn.MLP(
                in_size=force_in_size,
                out_size=3+num_genes+3,
                width_size=hidden_size,
                depth=num_layers,
                activation=jax.nn.swish,
                final_activation=lambda x: x,
                key=key_force,
            )   

    def __call__(self, t, X):
        N = X.shape[0]
        X_gene = X[:, 3:3+self.num_genes]
        if self.distinguish_range:
            raise NotImplementedError
            # X_morphogen_aggr_short, X_morphogen_aggr_long = self.aggr(t, X)     
            # input_short = jnp.concatenate((X_gene, X_morphogen_aggr_short), axis=-1)
            # input_long = jnp.concatenate((X_gene, X_morphogen_aggr_long), axis=-1)
            # if self.time_dependent:
            #     input_short = jnp.concatenate(
            #         (jnp.broadcast_to(t, (N, 1)), input_short),
            #         axis=-1,
            #     )
            #     input_long = jnp.concatenate(
            #         (jnp.broadcast_to(t, (N, 1)), input_long),
            #         axis=-1,
            #     )
            # F = 0.5 * (
            #     jax.vmap(self.force_short)(input_short) + 
            #     jax.vmap(self.force_long)(input_long)
            # )
        else:
            X_morphogen_aggr = self.aggr(t, X)
            input = jnp.concatenate((X_gene, X_morphogen_aggr), axis=-1)
            if self.time_dependent:
                input = jnp.concatenate(
                    (jnp.broadcast_to(t, (N, 1)), input),
                    axis=-1,
                )
            F = jax.vmap(self.force)(input)
        return F

class NeuralVectorField_e3nn(eqx.Module):
    aggr: AttentionAggregation
    weights: jnp.ndarray
    irreps_in: str
    irreps_out: str
    num_genes: int
    time_dependent: bool
    distinguish_range: bool
    use_rel_dist_pe: bool
    use_rel_angle_pe: bool

    ## Note that __init__ and __call__ in Equinox are equivalent
    ## to __init__ and forward methods from PyTorch

    ## Models are stored in a PyTree format
    ## which contains nested tuples, lists, and dictionaries
    ## (https://docs.kidger.site/equinox/all-of-equinox/#:~:text=Summary&text=Equinox%20introduces%20a%20powerful%20yet,else%20in%20the%20JAX%20ecosystem.)
    ## and are in this way compatible with JAX's JIT compilation and parallelization

    def __init__(self, 
        num_genes,
        num_morphogens, # size for QKV
        num_heads,
        hidden_size,
        num_layers,
        radius,
        time_dependent,
        distinguish_range,
        use_rel_dist_pe,
        use_rel_angle_pe,
        key,
    ):
        self.num_genes = num_genes
        self.time_dependent = time_dependent
        self.distinguish_range = distinguish_range
        self.use_rel_dist_pe = use_rel_dist_pe
        self.use_rel_angle_pe = use_rel_angle_pe
        
        key_attn, key_force = jrandom.split(key, 2)
        if distinguish_range:
            raise NotImplementedError(
                """We haven't made AttentionAggregation to distinguish
                short-ranged and long-ranged interaction.""")
        else:
            self.aggr = AttentionAggregation(
                in_features=num_genes,
                out_features=num_morphogens,
                num_heads=num_heads,
                time_dependent=time_dependent,
                use_rel_dist_pe=use_rel_dist_pe,
                use_rel_angle_pe=use_rel_angle_pe,
                key=key_attn,
            )

        ## Track the size of the input features,
        ## which is number of genes + number of morphogens,
        ## with an additional time dimension if time-dependent

        force_in_size = num_genes + num_morphogens
        if time_dependent:
            force_in_size += 1
            
        

        ### Defining, for input and output features,
        ## strings denoting their representations
        ### as the sum of irreducible representations (irreps).

        ### Recall that a representation maps
        ### each element of a group to a linear transformation in a vector space

        ## In this context,
        ### For each input (cell),
        ### Each gene, morphogen, and time step is a scalar (0e),
        ### and we wish to predict scalar updates to the expression (0e) and 
        ## forces (vectors, (1o)) in each of the 3 dimensions of space.

        ### Note the parity of each irrep, namely
        ### even (e) for scalar-valued gene / morphogen / time values, which wouldn't change sign upon inversion
        ### or odd (o) for vector-valued forces, which would change sign upon inversion
        
        ### Reference: https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html

        self.irreps_in = f"{force_in_size}x0e"  # force_in_size scalar inputs
        self.irreps_out = f"{num_genes}x0e + 3x1o"  # num_genes scalar outputs + 3 vector outputs
        
        ### Weight initialization, setting the seed (key) for reproducibility
        ### In this case, weights are normally distributed upon initialization
        self.weights = e3nn.normal(self.irreps_out, key=key_force) 
        ### ^ These weights will be used in a tensor product calculation in the __call__ method below,
        ### hence their irreps must match model output

    def __call__(self, t, X):
        N = X.shape[0]
        X_gene = X[:, 3:3+self.num_genes]
        if self.distinguish_range:
            raise NotImplementedError
        else:
            ### Attention-based morphogen aggregation
            X_morphogen_aggr = self.aggr(t, X)
            input = jnp.concatenate((X_gene, X_morphogen_aggr), axis=-1)
            if self.time_dependent:
                input = jnp.concatenate(
                    (jnp.broadcast_to(t, (N, 1)), input),
                    axis=-1,
                )
            
            ### IrrepsArray object from input
            input_irreps = e3nn.IrrepsArray(self.irreps_in, input)
            
            ### Calculate tensor product of input and weights
            F = e3nn.tensor_product(input_irreps, self.weights)
            
            ### Extract forces and gene expression values from the result / output.

            forces = F.array[:, -3:]  ### Vectors for forces in the last 3 columns
            gene_updates = F.array[:, :self.num_genes]  ### Scalar values for gene expression in the first num_genes columns
            
            output = X.copy()  ### The output will have the same arrangement as X
            output = output.at[:, :3].set(forces)  ### Set the forces at the next time step
            output = output.at[:, 3:3+self.num_genes].set(gene_updates)  ### Set the gene expression values at the next time step
            
        return output

class NeuralVectorField_e3nnForce_se3Attn(eqx.Module):
    aggr: AttentionAggregation_SE3_transformer
    weights: jnp.ndarray
    irreps_in: str
    irreps_out: str
    num_genes: int
    time_dependent: bool
    distinguish_range: bool
    use_rel_dist_pe: bool
    use_rel_angle_pe: bool

    def __init__(self, 
        num_genes,
        num_morphogens, # size for QKV
        num_heads,
        hidden_size,
        num_layers,
        radius,
        time_dependent,
        distinguish_range,
        use_rel_dist_pe,
        use_rel_angle_pe,
        key,
    ):
        self.num_genes = num_genes
        self.time_dependent = time_dependent
        self.distinguish_range = distinguish_range
        self.use_rel_dist_pe = use_rel_dist_pe
        self.use_rel_angle_pe = use_rel_angle_pe
        
        key_attn, key_force = jrandom.split(key, 2)
        if distinguish_range:
            raise NotImplementedError(
                """We haven't made AttentionAggregation to distinguish
                short-ranged and long-ranged interaction.""")
        else:
            self.aggr = AttentionAggregation(
                in_features=num_genes,
                out_features=num_morphogens,
                num_heads=num_heads,
                time_dependent=time_dependent,
                use_rel_dist_pe=use_rel_dist_pe,
                use_rel_angle_pe=use_rel_angle_pe,
                key=key_attn,
            )

        force_in_size = num_genes + num_morphogens
        if time_dependent:
            force_in_size += 1

        self.irreps_in = f"{force_in_size}x0e"  
        self.irreps_out = f"{num_genes}x0e + 3x1o"  
        
        self.weights = e3nn.normal(self.irreps_out, key=key_force)

    def __call__(self, t, X):
        N = X.shape[0]
        X_gene = X[:, 3:3+self.num_genes]
        if self.distinguish_range:
            raise NotImplementedError
        else:
            X_morphogen_aggr = self.aggr(t, X)
            input = jnp.concatenate((X_gene, X_morphogen_aggr), axis=-1)
            if self.time_dependent:
                input = jnp.concatenate(
                    (jnp.broadcast_to(t, (N, 1)), input),
                    axis=-1,
                )
    
            input_irreps = e3nn.IrrepsArray(self.irreps_in, input)
            
            F = e3nn.tensor_product(input_irreps, self.weights)
            forces = F.array[:, -3:]  
            gene_updates = F.array[:, :self.num_genes]  
    
            output = X.copy()  
            output = output.at[:, :3].set(forces)  
            output = output.at[:, 3:3+self.num_genes].set(gene_updates)  
            
            
        return output
