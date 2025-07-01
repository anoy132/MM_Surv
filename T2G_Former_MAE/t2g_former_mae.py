# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from models.T2G_Former_MAE import lib


# %%
class Tokenizer(nn.Module):
    """
    References:
    - FT-Transformer: https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py#L18
    """
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')

        # take [Cross-level Readout Node] into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )   # 把 [CLS] 拼接到 x_num上， [batch_size, num_feat_n] -> [batch_size, 1 + num_feat_n];
        x = self.weight[None] * x_num[:, :, None]   # [1, 1 + num_feat_n, d] x [batch_size, 1 + num_feat_n, 1] = [batch_size, 1 + num_feat_n, d]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadGEAttention(nn.Module):
    """
    FR-Graph integrated attention
    ---
    Learn relations among features and feature selection strategy in data-driven manner.

    """
    def __init__(
        # Normal Attention Args
        self, d: int, n_heads: int, dropout: float, initialization: str,
        # FR-Graph Args
        n: int, sym_weight: bool = True, sym_topology: bool = False, nsi: bool = True,
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        
        """FR-Graph Params: Edge weights"""
        # head and tail transformation
        self.W_head = nn.Linear(d, d)
        if sym_weight:
            self.W_tail = self.W_head # symmetric weights
        else:
            self.W_tail = nn.Linear(d, d) # ASYM
        # relation embedding: learnable diagonal matrix
        self.rel_emb = nn.Parameter(torch.ones(n_heads, d // self.n_heads))   

        for m in [self.W_head, self.W_tail, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

        """FR-Graph Params: Graph topology (column = node = feature)"""
        self.n_cols = n + 1 # Num of Nodes: input feature nodes + [Cross-level Readout]
        self.nsi = nsi # no self-interaction

        # column embeddings: semantics for each column
        d_col = math.ceil(2 * math.log2(self.n_cols)) # dim for column header embedding -> d_header += d
        self.col_head = nn.Parameter(Tensor(self.n_heads, self.n_cols, d_col))
        if not sym_topology:
            self.col_tail = nn.Parameter(Tensor(self.n_heads, self.n_cols, d_col))
        else:
            self.col_tail = self.col_head # share the parameter
        for W in [self.col_head, self.col_tail]:
            if W is not None:
                # correspond to Tokenizer initialization
                nn_init.kaiming_uniform_(W, a=math.sqrt(5))
        
        # Learnable bias and fixed threshold for topology
        self.bias = nn.Parameter(torch.zeros(1))
        self.threshold = 0.5

        """Frozen topology"""
        # for some sensitive datasets set to `True`
        # after training several epoch, which helps
        # stability and better performance
        self.frozen = False


    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
        )
    
    def _no_self_interaction(self, x):
        if x.shape[-2] == 1: # only [Readout Node]
            return x
        assert x.shape[-1] == x.shape[-2]
        current_col_num = x.shape[-1]
        # mask diagonal interaction
        nsi_mask = 1.0 - torch.diag_embed(torch.ones(current_col_num, device=x.device))
        return x * nsi_mask
    
    def _prune_to_readout(self, x):
        """Prune edges from any features to [Readout Node]"""
        # assert x.shape[-1] == self.n_cols
        mask = torch.ones(x.shape[-1], device=x.device)
        mask[0] = 0 # zero out interactions from features to [Readout]
        return x * mask
    
    def _get_topology(self, top_score, elewise_func=torch.sigmoid):
        """
        Learning static knowledge topology (adjacency matrix)
        ---
        top_score: N x N tensor, relation topology score
        adj: adjacency matrix A of FR-Graph
        """
        adj_probs = elewise_func(top_score + self.bias) # choose `sigmoid` as element-wise activation (sigma1)
        if self.nsi:
            adj_probs = self._no_self_interaction(adj_probs) # apply `nsi` function
        adj_probs = self._prune_to_readout(adj_probs) # cut edges from features to [Readout] # 直接将[CLS]和其他特征的关联变成0；
        
        if not self.frozen:
            # using `Straight-through` tirck for non-differentiable operation
            adj = (adj_probs > 0.5).float() - adj_probs.detach() + adj_probs   # 这里是一个binary mask;
        else:
            # frozen graph topology: no gradient
            adj = (adj_probs > 0.5).float()
        return adj

    def forward(
        self,
        x_head: Tensor,
        x_tail: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
        elewise_func = torch.sigmoid,
        comp_func = torch.softmax,
        enc:bool=False,
        dec:bool=False,
        mask = None,
        ids_keep=None,
        mask_embedding=None
    ) -> Tensor:
        assert enc ^ dec

        f_head, f_tail, f_v = self.W_head(x_head), self.W_tail(x_tail), self.W_v(x_tail)
        for tensor in [f_head, f_tail, f_v]:
            # check multi-head
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            f_tail = key_compression(f_tail.transpose(1, 2)).transpose(1, 2)
            f_v = value_compression(f_v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(f_head)
        d_head_tail = f_tail.shape[-1] // self.n_heads
        d_value = f_v.shape[-1] // self.n_heads
        n_head_nodes = f_head.shape[1]

        # reshape to multi-head view
        f_head = self._reshape(f_head)   # [batch_size, head_num, 1+feat_n OR 1, feat_dim_per_head];
        f_tail = self._reshape(f_tail)

        # edge weight scores (Gw)   [batch_size, head_num, 1+feat_n OR 1, 1+feat_n]
        weight_score = f_head @ torch.diag_embed(self.rel_emb) @ f_tail.transpose(-1, -2) / math.sqrt(d_head_tail)
        
        col_emb_head = F.normalize(self.col_head, p=2, dim=-1) # L2 normalized column embeddings
        col_emb_tail = F.normalize(self.col_tail, p=2, dim=-1)

        if enc and (mask is not None):
            assert ids_keep is not None

            col_emb_head_cls = col_emb_head[:,:1,:]   # [head_num, 1, dim]
            col_emb_head = col_emb_head[:,1:,:]
            col_emb_head = col_emb_head.unsqueeze(0).expand(batch_size,-1,-1,-1)
            ids_keep_expand = ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, self.col_head.shape[0], -1, self.col_head.shape[-1])
            col_emb_head = torch.gather(col_emb_head, dim=2, index=ids_keep_expand)  # [batch_size, head_num, keep_num, dim];
            col_emb_head_cls = col_emb_head_cls.unsqueeze(dim=0).expand(col_emb_head.shape[0],-1,-1,-1)
            col_emb_head = torch.concat([col_emb_head_cls, col_emb_head], dim=2)

            col_emb_tail_cls = col_emb_tail[:,:1,:]   # [head_num, 1, dim]
            col_emb_tail = col_emb_tail[:,1:,:]
            col_emb_tail = col_emb_tail.unsqueeze(0).expand(batch_size,-1,-1,-1)
            ids_keep_expand = ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, self.col_tail.shape[0], -1, self.col_tail.shape[-1])
            col_emb_tail = torch.gather(col_emb_tail, dim=2, index=ids_keep_expand)  # [batch_size, head_num, keep_num, dim];
            col_emb_tail_cls = col_emb_tail_cls.unsqueeze(dim=0).expand(col_emb_tail.shape[0],-1,-1,-1)
            col_emb_tail = torch.concat([col_emb_tail_cls, col_emb_tail], dim=2)

        # topology score (Gt)
        # 如果用了mask，应该是 [batch_size, head_num, 1+keep_num, 1+keep_num];
        top_score = col_emb_head @ col_emb_tail.transpose(-1, -2)   # [head_num, 1+feat_n, 1+feat_n];
        # graph topology (A)
        adj = self._get_topology(top_score, elewise_func)
        if n_head_nodes == 1: # only [Cross-level Readout]
            adj = adj[:, :1]
        
        # graph assembling: apply FR-Graph on interaction like attention mask
        adj_mask = (1.0 - adj) * -10000 # analogous to attention mask  # [head_num, 1+feat_n OR 1, 1+feat_n]

        # if mask is not None: # [batch_size, feat_n], 1 for remove;
        #     assert mask.shape[-1] == self.n_cols - 1
        #     mask_with_cls = torch.concat(
        #         [torch.zeros(mask.shape[0], 1, device=mask.device), mask], dim=-1
        #     )
        #     mask_with_cls = 1 - mask_with_cls   # [batch_size, 1 + feat_n], 1 for keep
        #     mask_with_cls = mask_with_cls.unsqueeze(-1) @ mask_with_cls.unsqueeze(1)  # [batch_size, 1 + feat_n, 1 + feat_n], 1 for keep
            

        # FR-Graph of this layer  [batch_size, head_num, 1+feat_n OR 1, 1+feat_n]
        # Can be used for visualization on Feature Relation and Readout Collection
        fr_graph = comp_func(weight_score + adj_mask, dim=-1) # choose `softmax` as competitive function


        if self.dropout is not None:
            fr_graph = self.dropout(fr_graph)
        x = fr_graph @ self._reshape(f_v)   # [batch_size, head_num, 1+feat_n OR 1, feat_dim_per_head];
        x = (
            x.transpose(1, 2)
            .reshape(batch_size, n_head_nodes, self.n_heads * d_value)
        )  # [batch_size, 1+feat_n OR 1, d_token];
        if self.W_out is not None:
            x = self.W_out(x)
        return x, fr_graph.detach()


class T2GFormer_MAE(nn.Module):

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        n_layers_dec:int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        # graph estimator
        sym_weight: bool = True,
        sym_topology: bool = False,
        nsi: bool = True,
        #
        d_out: int,
        # MAE
        mask_ratio:float=None,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)
        self.sym_topology = sym_topology

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens
        self.d_numerical = d_numerical

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        n_tokens = d_numerical if categories is None else d_numerical + len(categories)   # 特征数量，num+cat;
        d_hidden = int(d_token * d_ffn_factor)

        # encoder;
        self.encoder = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadGEAttention(
                        d_token, n_heads, attention_dropout, initialization,
                        n_tokens, sym_weight=sym_weight, sym_topology=sym_topology, nsi=nsi,
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.encoder.append(layer)

        # decoder;
        self.decoder = nn.ModuleList([])
        for layer_idx in range(n_layers_dec):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadGEAttention(
                        d_token, n_heads, attention_dropout, initialization,
                        n_tokens, sym_weight=sym_weight, sym_topology=sym_topology, nsi=nsi,
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.decoder.append(layer)

        self.activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

        self.mask_ratio = mask_ratio
        self.mask_embedding_encoder = nn.Linear(n_layers * math.ceil(2 * math.log2(n_tokens + 1)) * (1 if sym_topology else 2),
                                                d_token // n_heads)
        self.decoder_pred_list = nn.ModuleList([])
        for cat_sub_num in categories:
            self.decoder_pred_list.append(
                nn.Linear(d_token, cat_sub_num)
            )

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        References:
        facebook MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py#L123
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep
    
    def forward_encoder(self, x_num, x_cat, return_fr, mask_flag:bool=True):
        fr_graphs = [] # FR-Graph of each layer
        x = self.tokenizer(x_num, x_cat)   # [batch_size, 1 + num_feat_n + cat_feat_n, d_token];
        input_x_tokenized = x

        cls_token = x[:,:1,:]

        mask = ids_keep = None
        if mask_flag:
            x_masked, mask, ids_restore, ids_keep = self.random_masking(x[:,1:,:], mask_ratio=self.mask_ratio)
            x_masked = torch.concat([cls_token, x_masked], dim=1)
            x = x_masked

        for layer_idx, layer in enumerate(self.encoder):
            # is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual, fr_graph = layer['attention'](
                x_residual, x_residual,
                *self._get_kv_compressions(layer),
                enc=True,
                mask=mask, ids_keep=ids_keep,
            )
            fr_graphs.append(fr_graph)
            # if is_last_layer:
            #     x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        if mask_flag:
            return x, mask, ids_keep, ids_return
        else:
            return x
    
    def construct_masked_embedding(self):
        '''
        gather col feat from each layer of encoder, encode using linear layer, use as masked embedding;
        '''
        col_feat_list = []
        for layer in self.encoder:
            col_feat_list.append(F.normalize(layer['attention'].col_head[:,1:,:], p=2, dim=-1))
            if not self.sym_topology:
                col_feat_list.append(F.normalize(layer['attention'].col_tail[:,1:,:], p=2, dim=-1))
        col_feat_cat = torch.cat(col_feat_list, -1)   # [head_num, L, head_dim * layer_num];
        col_feat_cat = self.mask_embedding_encoder(col_feat_cat)   # [head_num, L, head_dim];
        keep_num = col_feat_cat.shape[1]
        col_feat_cat = col_feat_cat.transpose(1,0).reshape(keep_num, -1)   # [L, dim]
        return col_feat_cat
    
    def forward_decoder(self, x, mask, ids_keep, ids_restore):
        batch_size = x.shape[0]
        # x is like [batch_size, keep_num, dim],
        masked_embedding = self.construct_masked_embedding()
        masked_embedding = masked_embedding.unsqueeze(0).expand(batch_size, -1, -1).to(x.dtype)

        cls_token = x[:,:1,:]
        x_wo_cls = x[:,1:,:]

        x_wo_cls = torch.gather(x_wo_cls, dim=1, index=ids_restore.unsqueeze(-1).repeat(-1, -1, x.shape[-1]))

        ids_masked = torch.nonzero(mask, as_tuple=True)[1].view(batch_size, -1)

        # Step 2: Retrieve Masked Features from `feature_map`
        retrieved_features = torch.gather(masked_embedding, dim=1, index=ids_masked.unsqueeze(-1).expand(-1, -1, masked_embedding.shape[-1]))

        # Step 3: Reconstruct Full Sequence
        full_reconstructed = torch.zeros((batch_size, masked_embedding.shape[1], masked_embedding.shape[-1]),  
                                         device=x.device)  # Initialize empty tensor
        # Insert unmasked tokens
        full_reconstructed.scatter_(dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, masked_embedding.shape[-1]), src=x_wo_cls)
        # Insert retrieved masked tokens
        full_reconstructed.scatter_(dim=1, index=ids_masked.unsqueeze(-1).expand(-1, -1, masked_embedding.shape[-1]), src=retrieved_features)

        x = torch.concat([cls_token, full_reconstructed], dim=1)

        for layer_idx, layer in enumerate(self.decoder):
            # is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual, fr_graph = layer['attention'](
                x_residual, x_residual,
                *self._get_kv_compressions(layer),
                dec=True,
                mask=mask, ids_keep=ids_keep
            )
            # if is_last_layer:
            #     x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        return x
    

    # def forward_loss(self, x, x_num, x_cat, mask):
    #     cls_token = x[:,:1,:]
    #     x = x[:,1:,:]
    #     assert mask.shape[1] == x.shape[1]

    #     if self.last_normalization is not None:
    #         x = self.last_normalization(x)
    #     x = self.last_activation(x)

    #     pred_num = x[:,:self.d_numerical,:]
    #     pred_num = self.head(pred_num).squeeze(-1)
    #     num_loss = F.mse_loss(pred_num, x_num, reduction='none')   # [batch_size, num_feat_n];
    #     num_mask = mask[:,:self.d_numerical]
    #     num_loss = torch.sum(num_loss * num_mask) / torch.sum(num_mask)

    #     total_loss = num_loss

    #     cat_mask = mask[:,self.d_numerical:]
    #     pred_cat = x[:,self.d_numerical:,:]
    #     for idx, sub_module in enumerate(self.decoder_pred_list):
    #         sub_pred_cat = sub_module(pred_cat[:,idx,:])
    #         assert sub_pred_cat.shape[-1] >=2
    #         current_cat_mask = cat_mask[:,idx]
    #         if sub_pred_cat.shape[-1] == 2:
    #             gt = torch.stack([x_cat[:,idx], 1-x_cat[:,idx]], dim=-1).float()
    #             sub_pred_cat = torch.clamp(sub_pred_cat, 1e-8)
    #             cat_loss = F.binary_cross_entropy_with_logits(sub_pred_cat, gt).sum(-1)
    #             cat_loss = torch.sum(cat_loss * current_cat_mask) / torch.sum(current_cat_mask)
    #             total_loss = total_loss + cat_loss
    #         elif sub_pred_cat.shape[-1] > 2:
    #             cat_loss = F.cross_entropy(sub_pred_cat, x_cat[:,idx], reduction='none')
    #             cat_loss = torch.sum(cat_loss * current_cat_mask) / torch.sum(current_cat_mask)
    #             total_loss = total_loss + cat_loss
        
    #     return total_loss


    def forward_loss(self, x, x_num, x_cat, mask):
        cls_token = x[:,:1,:]
        x = x[:,1:,:]
        assert mask.shape[1] == x.shape[1]

        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)

        pred = self.head(x).squeeze(-1)  # [feat_num, 1];
        gt = torch.concat([x_num, x_cat], dim=-1)
        temp = (pred - gt) **2
        total_loss = (temp * mask).sum() / mask.sum()
        
        return total_loss
    

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], return_fr: bool = False) -> Tensor:

        latent, masks, ids_keep, ids_restore = self.forward_encoder(x_num, x_cat, return_fr)
        pred = self.forward_decoder(latent, masks, ids_keep, ids_restore)
        loss = self.forward_loss(pred, x_num, x_cat, masks)
        return loss
    
    def froze_topology(self):
        """API to froze FR-Graph topology in training"""
        for layer in self.encoder:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            layer['attention'].frozen = True

        for layer in self.decoder:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            layer['attention'].frozen = True
