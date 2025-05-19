import torch
import torch.nn as nn



# class Surv_MLP(nn.Module):
#     def __init__(self, input_dim, layer_num, output_dim=1, intermediate_size=None):
#         super().__init__()
#         self.enc = nn.ModuleList()
#         for _ in range(layer_num):
#             self.enc.append(
#                 nn.Sequential(
#                     LlamaMLP_Residual(input_dim, intermediate_size),
#                 LayerNorm(input_dim)
#                 )
#             )
#         self.enc.append(
#             nn.Linear(input_dim, output_dim, bias=True)
#         )


#     def forward(self, x):
#         for layer in self.enc:
#             x = layer(x)
#         return x
    

class Surv_MLP(nn.Module):
    def __init__(self, input_dim, layer_num, output_dim=1, intermediate_size=None, dropout=0.0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, intermediate_size or input_dim),
            nn.ReLU(),
            # nn.SiLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(intermediate_size or input_dim, intermediate_size or input_dim),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(intermediate_size or input_dim, intermediate_size or input_dim),
            nn.Linear(intermediate_size or input_dim, output_dim)
        )

        # self.enc = nn.Sequential(
        #     nn.Linear(input_dim, input_dim//2),
        #     # nn.ReLU(),
        #     nn.SiLU(),
        #     nn.Dropout(p=0.1),
        #     # nn.Linear(intermediate_size or input_dim, intermediate_size or input_dim),
        #     # nn.ReLU(),
        #     # nn.Dropout(p=0.1),
        #     # nn.Linear(intermediate_size or input_dim, intermediate_size or input_dim),
        #     nn.Linear(input_dim//2, output_dim)
        # )

        # self.enc = nn.Sequential(
        #     nn.Linear(input_dim, intermediate_size or input_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(intermediate_size or input_dim, intermediate_size or input_dim),
        #     nn.ReLU(),
        #     nn.Linear(intermediate_size or input_dim, output_dim),
        # )


    def forward(self, x):
        x = self.enc(x)
        return x


# class Surv_MLP(nn.Module):
#     def __init__(self, input_dim, layer_num, output_dim=1, intermediate_size=None):
#         super().__init__()
#         self.in_enc_0 = nn.Linear(input_dim//2, (intermediate_size or input_dim)//2)
#         self.in_enc_1 = nn.Linear(input_dim//2, (intermediate_size or input_dim)//2)

#         self.enc = nn.Sequential(
#             nn.Linear(input_dim, intermediate_size or input_dim),
#             # nn.ReLU(),
#             nn.SiLU(),
#             nn.Dropout(p=0.2),
#             # nn.Linear(intermediate_size or input_dim, intermediate_size or input_dim),
#             # nn.ReLU(),
#             # nn.Dropout(p=0.1),
#             # nn.Linear(intermediate_size or input_dim, intermediate_size or input_dim),
#             nn.Linear(intermediate_size or input_dim, output_dim)
#         )

#     def forward(self, x):
#         c_dim = x.shape[-1]
#         x_0 = self.in_enc_0(x[:,:c_dim//2])
#         x_1 = self.in_enc_0(x[:,c_dim//2:])
#         x = torch.concat([x_0,x_1],dim=-1)
#         x = self.enc(x)
#         return x




class LlamaMLP_Residual(nn.Module):
    def __init__(self, hidden_size, intermediate_size, output_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size
        self.output_size = output_size or hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        down_proj = x + down_proj
        return down_proj
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)