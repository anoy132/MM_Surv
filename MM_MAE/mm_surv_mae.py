import torch
import torch.nn as nn
import sys
sys.path.append('/home/xql/work/Experiment/Medical_image_analysis/experiment/XQL_medical_image_arch/TongjiMultiModal/handcrafts_xql/code/MultiModal_MAE/models')
sys.path.append('/home/xql/work/Experiment/Medical_image_analysis/experiment/XQL_medical_image_arch/TongjiMultiModal/handcrafts_xql/code/MultiModal_MAE/models/MM_MAE/t2g_former_utils')
from CT_CLIP import CTViT_MAE
from T2G_Former_MAE import T2GFormer_MAE, T2GFormer_MAE_LossV1
from .detr_transformer import TransformerDecoderLayer, TransformerDecoder
from transformers import BertTokenizer, BertModel
from einops import rearrange
import torch.nn.functional as F



class MM_Surv_MAE(nn.Module):

    def __init__(self, text_cfg, vis_cfg, tab_cfg, mid_cfg, 
                 surv_loss_func,
                 loss_cfg=None):

        super().__init__()
        
        self.tokenizer, self.text_encoder = self.get_text_model(text_cfg, vis_cfg)
        self.vis_model = self.get_vis_model(vis_cfg)
        self.tab_model = self.get_tab_model(tab_cfg)

        self.mid_vis_module = self.get_mid_vis_module(mid_cfg, tab_cfg, vis_cfg)
        self.mid_tab_module = self.get_mid_tab_module(mid_cfg, tab_cfg, vis_cfg)

        self.loss_cfg = loss_cfg

        self.surv_pred = nn.Sequential(
            nn.Linear(tab_cfg.get('d_token')+vis_cfg.get('dim', 512),
                      tab_cfg.get('d_token')+vis_cfg.get('dim', 512)),
            nn.SiLU(),
            nn.Linear(tab_cfg.get('d_token')+vis_cfg.get('dim', 512), 1)
        )
        self.surv_loss_func = surv_loss_func


    def get_text_model(self, text_cfg, vis_cfg):
        tokenizer = BertTokenizer.from_pretrained(text_cfg['bert_path'], do_lower_case=True)
        text_encoder = BertModel.from_pretrained(text_cfg['bert_path'])
        text_encoder.resize_token_embeddings(len(tokenizer))

        self.text_to_latent = nn.Linear(text_cfg.get('dim', 768), vis_cfg.get('dim', 512), 
                                        bias=text_cfg.get('to_latent_bias', False))
        if text_cfg.get('pretrained_ckpt', None) is not None:
            # 这里假设使用的是CT-CLIP里面那个
            pretrained_ckpt = torch.load(text_cfg.get('pretrained_ckpt'))
            temp = dict()
            for k,v in pretrained_ckpt.items(): 
                if 'to_text_latent.' in k:
                    if k in temp:
                        raise ValueError(f"Duplicate keys found: {k}")
                    else:
                        temp.update({str.split(k, 'to_text_latent.')[1]:v})
            self.text_to_latent.load_state_dict(temp, strict=True)
            if not text_cfg.get('text_to_latent_trainable', False):
                for param in self.text_to_latent.parameters():
                    param.requires_grad = False  # Freezes the layer

        return tokenizer, text_encoder
    

    def get_vis_model(self, vis_cfg):
        model = CTViT_MAE(
            dim = vis_cfg.get('dim', 512),
            codebook_size = vis_cfg.get('codebook_size', 8192),
            image_size = vis_cfg.get('image_size', 480),
            patch_size = vis_cfg.get('patch_size', 20),
            temporal_patch_size = vis_cfg.get('temporal_patch_size', 10),
            spatial_depth = vis_cfg.get('spatial_depth', 4),
            temporal_depth = vis_cfg.get('temporal_depth', 4),
            dim_head = vis_cfg.get('dim_head', 32),
            heads = vis_cfg.get('heads', 8),
            mask_ratio=vis_cfg.get('mask_ratio', 0.5),
            temporal_size=vis_cfg.get('temporal_size', 240)
            )
        
        pretrained_ckpt = vis_cfg.get('pretrained_ckpt', None)
        if pretrained_ckpt is not None:
            pretrained_ckpt = torch.load(pretrained_ckpt, map_location="cpu")
            model.load_state_dict(pretrained_ckpt, strict=False)
        
        return model
    

    def get_tab_model(self, tab_cfg):
        model = T2GFormer_MAE(
            **tab_cfg
        )
        return model
    

    def get_mid_vis_module(self, mid_cfg, tab_cfg, vis_cfg):
        decoder_layer = TransformerDecoderLayer(vis_cfg.get('dim', 512),
                                                mid_cfg.vis.get('heads', 4) , 
                                                mid_cfg.vis.get('hidden_dim', 1024),
                                                mid_cfg.vis.get('dropout', 0.1),
                                                mid_cfg.vis.get('act', 'relu'),
                                                normalize_before=mid_cfg.vis.get('normalize_before', True)
                                                )
        fuse_module = TransformerDecoder(decoder_layer, 
                                    mid_cfg.vis.get('layer_num', None), 
                                    norm=nn.LayerNorm(vis_cfg.get('dim', 512)),
                                    return_intermediate=mid_cfg.vis.get('return_intermediate', False))
        init_module = nn.Linear(
            tab_cfg.get('d_token'), vis_cfg.get('dim', 512)
        )

        module_dict = nn.ModuleDict(
            {'init':init_module, 'fuse':fuse_module}
        )
        
        return module_dict
    

    def get_mid_tab_module(self, mid_cfg, tab_cfg, vis_cfg):
        decoder_layer = TransformerDecoderLayer(tab_cfg.get('d_token'),
                                                mid_cfg.tab.get('heads', 4) , 
                                                mid_cfg.tab.get('hidden_dim', 1024),
                                                mid_cfg.tab.get('dropout', 0.1),
                                                mid_cfg.tab.get('act', 'relu'),
                                                normalize_before=mid_cfg.tab.get('normalize_before', True)
                                                )
        fuse_module = TransformerDecoder(decoder_layer, 
                                    mid_cfg.tab.get('layer_num', None), 
                                    norm=nn.LayerNorm(tab_cfg.get('d_token')),
                                    return_intermediate=mid_cfg.tab.get('return_intermediate', False))

        init_module = nn.Linear(
            vis_cfg.get('dim', 512), tab_cfg.get('d_token')
        )
        
        module_dict = nn.ModuleDict(
            {'init':init_module, 'fuse':fuse_module}
        )

        return module_dict
    

    def fuse_tab_to_vis(self, tab_feat, vis_feat):
        assert vis_feat.ndim in {3,5}
        if vis_feat.ndim == 3:
            # batch和t合并了；
            b = tab_feat.shape[0]
            n = vis_feat.shape[1]
            vis_feat = rearrange(vis_feat, '(b t) n d -> (t n) b d', b=b)
            vis_feat_init_ndim = 3
        elif vis_feat.ndim == 5:
            vis_t, vis_h, vis_w = vis_feat.shape[1:-1]
            vis_feat = rearrange(vis_feat, 'b t h w d -> (t h w) b d')
            vis_feat_init_ndim = 5

        tab_feat = self.mid_vis_module['init'](tab_feat)
        tab_feat = rearrange(tab_feat, 'b l d -> l b d')
        fused_feat = self.mid_vis_module['fuse'](vis_feat, tab_feat)[0]

        if vis_feat_init_ndim == 5:
            fused_feat = rearrange(fused_feat, '(t h w) b d -> b t h w d', t=vis_t, h=vis_h, w=vis_w)
        elif vis_feat_init_ndim == 3:
            fused_feat = rearrange(fused_feat, '(t n) b d -> (b t) n d', n=n)

        return fused_feat
    

    def fuse_vis_to_tab(self, tab_feat, vis_feat):
        vis_feat = rearrange(vis_feat, 'b t h w d -> (t h w) b d')
        vis_feat = self.mid_tab_module['init'](vis_feat)
        tab_feat = rearrange(tab_feat, 'b l d -> l b d')
        fused_feat = self.mid_tab_module['fuse'](tab_feat, vis_feat)[0]
        fused_feat = rearrange(fused_feat, 'l b d -> b l d')
        return fused_feat
    

    def forward_encoder(self, sample):
        intact_vis_tokens, masked_vis_tokens, vis_mae_kwargs = \
        self.vis_model.forward_encoder(sample['image'], mask=None)

        masked_tab_tokens, tab_masks, tab_ids_keep = \
        self.tab_model.forward_encoder(sample['num_feat'], sample['cat_feat'], False, mask_flag=True)

        intact_tab_tokens = self.tab_model.forward_encoder(sample['num_feat'], sample['cat_feat'], False, mask_flag=False)
        tab_mae_kwargs = {
            'tab_masks':tab_masks, 'tab_ids_keep':tab_ids_keep
        }

        # 初始化一个loss字典；
        loss_dict = dict()

        return {
            'intact_vis_tokens':intact_vis_tokens,
            'masked_vis_tokens':masked_vis_tokens,
            'vis_mae_kwargs':vis_mae_kwargs,
            'intact_tab_tokens':intact_tab_tokens,
            'masked_tab_tokens':masked_tab_tokens,
            'tab_mae_kwargs':tab_mae_kwargs,
            'sample':sample,
            'loss_dict':loss_dict
        }
    

    def mid_interaction(self, storage:dict):
        def cos_sim(feat1,feat2):
            # Normalize both feature maps along dim=1 (feature dimension)
            feat1_norm = F.normalize(feat1, p=2, dim=1)  # L2 normalization
            feat2_norm = F.normalize(feat2, p=2, dim=1)
            # Compute cosine similarity along the feature dimension
            cos_sim = (feat1_norm * feat2_norm).sum(dim=1, keepdim=True)
            cos_sim = torch.clamp(cos_sim, 0, 1)
            return cos_sim
        
        def dist_loss(pos,neg):
            # cat = torch.cat([pos,neg], dim=1)   # [b,2];
            # gt = torch.zeros([cat.shape[0]], device=cat.device).long()
            # return F.cross_entropy(cat, gt)
            loss = - pos / (pos + neg + 1e-7)
            return loss.mean()
        
        device = storage['sample']['image'].device

        pos_prompt = list(storage['sample']['pos_prompt'])
        pos_prompt = self.tokenizer(pos_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        pos_prompt = self.text_encoder(pos_prompt.input_ids, attention_mask = pos_prompt.attention_mask)
        pos_prompt = self.text_to_latent(pos_prompt[0][:,0,:])   # [b,d];

        neg_prompt = list(storage['sample']['neg_prompt'])
        neg_prompt = self.tokenizer(neg_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        neg_prompt = self.text_encoder(neg_prompt.input_ids, attention_mask = neg_prompt.attention_mask)
        neg_prompt = self.text_to_latent(neg_prompt[0][:,0,:])


        ### vis part ###
        assert storage['intact_tab2intact_vis'].ndim == 5
        # intact_tab2intact_vis_pooled shape [b,d];
        intact_tab2intact_vis_pooled = torch.mean(storage['intact_tab2intact_vis'], 
                                                  dim=list(range(1, 4)))
        ii_pos_cos = cos_sim(intact_tab2intact_vis_pooled, pos_prompt)
        ii_neg_cos = cos_sim(intact_tab2intact_vis_pooled, neg_prompt)
        
        t = storage['intact_tab2intact_vis'].shape[1]
        assert storage['intact_tab2masked_vis'].ndim == 3
        intact_tab2masked_vis = rearrange(storage['intact_tab2masked_vis'], '(b t) n d -> b (t n) d', t=t)
        intact_tab2masked_vis_pooled = torch.mean(intact_tab2masked_vis, dim=1)
        im_pos_cos = cos_sim(intact_tab2masked_vis_pooled, pos_prompt)
        im_neg_cos = cos_sim(intact_tab2masked_vis_pooled, neg_prompt)

        # FIXME: 尝试使用对比学习；
        # 这里我用了一个detach来切断全intact的梯度回传；这个是类似于contrastive-based MIM的方法；
        # vis_sim_loss = (torch.mean((ii_pos_cos.detach() - im_pos_cos)**2) + torch.mean((ii_neg_cos.detach() - im_neg_cos)**2)) / 2
        ii_cos = torch.concat([ii_pos_cos, ii_neg_cos], dim=-1).unsqueeze(dim=1).detach()
        im_cos = torch.concat([im_pos_cos, im_neg_cos], dim=-1).unsqueeze(dim=0)
        temp = torch.mean((im_cos - ii_cos)**2, dim=-1)  # [b,b];
        temp_diag = torch.diagonal(temp).unsqueeze(-1)
        vis_sim_loss = torch.mean(temp_diag / (torch.sum(temp, -1, keepdim=True) + 1e-7))

        vis_intact_dist_loss = dist_loss(ii_pos_cos, ii_neg_cos)
        vis_masked_dist_loss = dist_loss(im_pos_cos, im_neg_cos)
        vis_total_dist_loss = 0.5 * (vis_intact_dist_loss + vis_masked_dist_loss)

        ### tab part ###
        intact_vis2intact_tab_cls = storage['intact_vis2intact_tab'][:,0,:]
        intact_vis2masked_tab_cls = storage['intact_vis2masked_tab'][:,0,:]
        ii_pos_cos = cos_sim(intact_vis2intact_tab_cls, pos_prompt)
        ii_neg_cos = cos_sim(intact_vis2intact_tab_cls, neg_prompt)
        im_pos_cos = cos_sim(intact_vis2masked_tab_cls, pos_prompt)
        im_neg_cos = cos_sim(intact_vis2masked_tab_cls, neg_prompt)

        # tab_sim_loss = (torch.mean((ii_pos_cos.detach() - im_pos_cos)**2) + torch.mean((ii_neg_cos.detach() - im_neg_cos)**2)) / 2
        ii_cos = torch.concat([ii_pos_cos, ii_neg_cos], dim=-1).unsqueeze(dim=1).detach()
        im_cos = torch.concat([im_pos_cos, im_neg_cos], dim=-1).unsqueeze(dim=0)
        temp = torch.mean((im_cos - ii_cos)**2, dim=-1)  # [b,b];
        temp_diag = torch.diagonal(temp).unsqueeze(-1)
        tab_sim_loss = torch.mean(temp_diag / (torch.sum(temp, -1, keepdim=True) + 1e-7))

        tab_intact_dist_loss = dist_loss(ii_pos_cos, ii_neg_cos)
        tab_masked_dist_loss = dist_loss(im_pos_cos, im_neg_cos)
        tab_total_dist_loss = 0.5 * (tab_intact_dist_loss + tab_masked_dist_loss)


        storage['loss_dict'].update(
            {'vis_sim_loss':vis_sim_loss,
             'vis_intact_dist_loss':vis_intact_dist_loss,
             'vis_masked_dist_loss':vis_masked_dist_loss,
             'vis_total_dist_loss':vis_total_dist_loss,
             'tab_sim_loss':tab_sim_loss,
            'tab_intact_dist_loss':tab_intact_dist_loss,
            'tab_masked_dist_loss':tab_masked_dist_loss,
            'tab_total_dist_loss':tab_total_dist_loss
             })

        return storage    
    

    def forward_mid(self, storage):
        intact_tab2intact_vis = self.fuse_tab_to_vis(tab_feat=storage['intact_tab_tokens'], vis_feat=storage['intact_vis_tokens'])
        intact_vis2intact_tab = self.fuse_vis_to_tab(tab_feat=storage['intact_tab_tokens'], vis_feat=storage['intact_vis_tokens'])

        intact_tab2masked_vis = self.fuse_tab_to_vis(tab_feat=storage['intact_tab_tokens'], vis_feat=storage['masked_vis_tokens'])
        intact_vis2masked_tab = self.fuse_vis_to_tab(tab_feat=storage['masked_tab_tokens'], vis_feat=storage['intact_vis_tokens'])

        storage.update(
            {'intact_tab2masked_vis':intact_tab2masked_vis,
             'intact_vis2masked_tab':intact_vis2masked_tab,
             'intact_tab2intact_vis':intact_tab2intact_vis,
             'intact_vis2intact_tab':intact_vis2intact_tab}
        )

        storage = self.mid_interaction(storage)

        return storage
    

    def forward_decoder(self, storage):
        recon_image, vis_commit_loss = self.vis_model.forward_decoder(None, storage['intact_tab2masked_vis'], 
                                                                      video=storage['sample']['image'], 
                                                                      mae_kwargs=storage['vis_mae_kwargs'])
        
        tab_recon = self.tab_model.forward_decoder(storage['intact_vis2masked_tab'], 
                                                   storage['tab_mae_kwargs']['tab_masks'], 
                                                   storage['tab_mae_kwargs']['tab_ids_keep'])
        storage.update(
            {'recon_image':recon_image, 'tab_recon':tab_recon}
        )

        storage['loss_dict'].update({'vis_commit_loss':vis_commit_loss})

        return storage
        

    def forward_loss(self, storage):
        vis_recon_loss = self.vis_model.forward_loss(storage['sample']['image'], 
                                                     storage['recon_image'], 
                                                     storage['vis_mae_kwargs']['mask'])
        
        # FIXME: 这个表格重建损失可能还要再调整一下；
        tab_recon_loss = self.tab_model.forward_loss(storage['tab_recon'], 
                                                     storage['sample']['num_feat'],
                                                     storage['sample']['cat_feat'],
                                                     storage['tab_mae_kwargs']['tab_masks'])
        vis_feat_pool = torch.mean(storage['intact_tab2intact_vis'], dim=list(range(1, 4)))
        tab_feat_cls = storage['intact_vis2intact_tab'][:,0,:]
        surv_pred = self.surv_pred(torch.cat([vis_feat_pool, tab_feat_cls], dim=-1))
        surv_loss = self.surv_loss_func(surv_pred, 
                                        storage['sample']['surv_time'], storage['sample']['surv_event'])[0]

        storage['loss_dict'].update({
            'vis_recon_loss':vis_recon_loss, 'tab_recon_loss':tab_recon_loss,
            'surv_loss':surv_loss
        })

        storage.update({'surv_pred':surv_pred})

        loss = 0
        for k,v in self.loss_cfg.items():
            loss = loss + storage['loss_dict'][k] * v
            storage['loss_dict'][k] = storage['loss_dict'][k] * v
            
        storage['loss_dict']['loss'] = loss

        loss_dict = {k.replace("tab_", "tab/").replace("vis_", "vis/"): v for k, v in storage['loss_dict'].items()}
        storage['loss_dict'] = loss_dict

        return storage
    

    def before_output(self, storage):
        new_dict = dict()
        new_dict['loss_dict'] = storage['loss_dict']
        if 'surv_pred' in storage:
            new_dict['surv_pred'] = storage['surv_pred']
        return new_dict
        

    def forward(self, sample):
        output = self.forward_encoder(sample)
        output = self.forward_mid(output)
        output = self.forward_decoder(output)
        output = self.forward_loss(output)
        output = self.before_output(output)

        return output




class MM_Surv_MAE_V1(MM_Surv_MAE):

    def __init__(self, text_cfg, vis_cfg, tab_cfg, mid_cfg, 
                    surv_loss_func, loss_cfg=None):
        super().__init__(text_cfg, vis_cfg, tab_cfg, mid_cfg, 
                         surv_loss_func, loss_cfg)


    def mid_interaction(self, storage:dict):
        def cos_sim(feat1,feat2):
            # Normalize both feature maps along dim=1 (feature dimension)
            feat1_norm = F.normalize(feat1, p=2, dim=1)  # L2 normalization
            feat2_norm = F.normalize(feat2, p=2, dim=1)
            # Compute cosine similarity along the feature dimension
            # cos_sim = torch.einsum('ab,cd -> ac', feat1_norm, feat2_norm)
            cos_sim = feat1_norm.mm(feat2_norm.t())
            return cos_sim
        
        device = storage['sample']['image'].device
        batch_size = storage['sample']['image'].shape[0]

        ### vis part ###
        assert storage['intact_tab2intact_vis'].ndim == 5
        # intact_tab2intact_vis_pooled shape [b,d];
        intact_tab2intact_vis_pooled = torch.mean(storage['intact_tab2intact_vis'], 
                                                  dim=list(range(1, 4)))
        
        t = storage['intact_tab2intact_vis'].shape[1]
        assert storage['intact_tab2masked_vis'].ndim == 3
        intact_tab2masked_vis = rearrange(storage['intact_tab2masked_vis'], '(b t) n d -> b (t n) d', t=t)
        intact_tab2masked_vis_pooled = torch.mean(intact_tab2masked_vis, dim=1)

        vis_sim = cos_sim(intact_tab2masked_vis_pooled, intact_tab2intact_vis_pooled.detach())
        vis_gt = torch.arange(batch_size).long().to(vis_sim.device)
        vis_sim_loss = F.cross_entropy(vis_sim, vis_gt)


        ### tab part ###
        intact_vis2intact_tab_cls = storage['intact_vis2intact_tab'][:,0,:]
        intact_vis2masked_tab_cls = storage['intact_vis2masked_tab'][:,0,:]

        tab_sim = cos_sim(intact_vis2masked_tab_cls, intact_vis2intact_tab_cls.detach())
        tab_gt = torch.arange(batch_size).long().to(tab_sim.device)
        tab_sim_loss = F.cross_entropy(tab_sim, tab_gt)


        storage['loss_dict'].update(
            {'vis_sim_loss':vis_sim_loss,
             'vis_intact_dist_loss':torch.zeros_like(vis_sim_loss),
             'vis_masked_dist_loss':torch.zeros_like(vis_sim_loss),
             'vis_total_dist_loss':torch.zeros_like(vis_sim_loss),
             'tab_sim_loss':tab_sim_loss,
            'tab_intact_dist_loss':torch.zeros_like(vis_sim_loss),
            'tab_masked_dist_loss':torch.zeros_like(vis_sim_loss),
            'tab_total_dist_loss':torch.zeros_like(vis_sim_loss)
             })

        return storage      
    


class MM_Surv_MAE_V2(MM_Surv_MAE_V1):

    def __init__(self, text_cfg, vis_cfg, tab_cfg, mid_cfg, 
                    surv_loss_func, loss_cfg=None):
        super().__init__(text_cfg, vis_cfg, tab_cfg, mid_cfg, 
                         surv_loss_func, loss_cfg)
    

    def get_tab_model(self, tab_cfg):
        model = T2GFormer_MAE_LossV1(
            **tab_cfg
        )
        return model




if __name__ == "__main__":
    import argparse
    import json
    import numpy as np
    from pathlib import Path
    from t2g_former_utils.lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer
    from torch.utils.data import Dataset, DataLoader
    from omegaconf import OmegaConf


    T2G_PROJ = Path('/home/xql/disk_1/work/Dataset/T2G_Former_Tabular_Datasets').absolute().resolve()
    T2G_EXP = T2G_PROJ / 'exp'
    T2G_DATA = T2G_PROJ / 'data'

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='adult')
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--t2g_model", type=str, default='T2GFormer_MAE')
    args = parser.parse_args()
    tab_cfg_file = f'/home/xql/work/Experiment/Medical_image_analysis/OpensourceCode/T2G_Former/configs/{args.dataset}/{args.t2g_model}/cfg.json'
    with open(tab_cfg_file, 'r') as f:
        tab_cfg = json.load(f)

    dataset_name = args.dataset
    T_cache = True
    normalization = args.normalization if args.normalization != '__none__' else None
    transformation = Transformations(normalization=normalization,
                                    #  cat_encoding='one-hot'
                                    )
    dataset = build_dataset(T2G_DATA / dataset_name, transformation, T_cache)

    if dataset.X_num['train'].dtype == np.float64:
        dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}

    d_out = dataset.n_classes or 1
    X_num, X_cat, ys = prepare_tensors(dataset, device=torch.device('cuda'))

    if dataset.task_type.value == 'regression':
        y_std = ys['train'].std().item()

    batch_size_dict = {
        'churn': 128, 'eye': 128, 'gesture': 128, 'california': 256, 'house': 256, 'adult': 256 , 
        'higgs-small': 512, 'helena': 512, 'jannis': 512, 'otto': 512, 'fb-comments': 512,
        'covtype': 1024, 'year': 1024, 'santander': 1024, 'microsoft': 1024, 'yahoo': 256}
    val_batch_size = 1024 if args.dataset in ['santander', 'year', 'microsoft'] else 256 if args.dataset in ['yahoo'] else 8192
    # val_batch_size = 64
    if args.dataset == 'epsilon':
        batch_size = 16 if args.dataset == 'epsilon' else 128 if args.dataset == 'yahoo' else 256
    elif args.dataset not in batch_size_dict:
        if dataset.n_features <= 32:
            batch_size = 512
            val_batch_size = 8192
        elif dataset.n_features <= 100:
            batch_size = 128
            val_batch_size = 512
        elif dataset.n_features <= 1000:
            batch_size = 32
            val_batch_size = 64
        else:
            batch_size = 16
            val_batch_size = 16
    else:
        batch_size = batch_size_dict[args.dataset]

    num_workers = 0
    data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]

    n_num_features = dataset.n_num_features
    cardinalities = dataset.get_category_sizes('train')   # 这个是查看各个类别中分别有几个小类，比如性别可能就是2,因为只有男女；
    n_categories = len(cardinalities)   
    cardinalities = None if n_categories == 0 else cardinalities

    """set default"""
    tab_cfg['model'].setdefault('kv_compression', None)
    tab_cfg['model'].setdefault('kv_compression_sharing', None)
    tab_cfg['model'].setdefault('token_bias', True)
    # default FR-Graph settings
    tab_cfg['model'].setdefault('sym_weight', True)
    tab_cfg['model'].setdefault('sym_topology', False)
    tab_cfg['model'].setdefault('nsi', True)
    """prepare model arguments"""
    kwargs = {
        # task related
        'd_numerical': n_num_features,
        'categories': cardinalities,
        'd_out': d_out,
        **tab_cfg['model']
    }
    kwargs['d_token'] = 512

    tab_model_cfg = OmegaConf.create(kwargs)

    text_model_cfg = OmegaConf.create(dict(bert_path='/home/xql/disk_1/work/LargeModels/LanguageModels/microsoft/BiomedVLP-CXR-BERT-specialized'))

    mid_model_cfg = OmegaConf.create({'vis':dict(layer_num=2), 'tab':dict(layer_num=2)})

    vis_cfg = OmegaConf.create(dict(patch_size=30, temporal_patch_size=15))

    loss_cfg = {'vis_sim_loss':1.0, 'vis_total_dist_loss':1.0, 
                'tab_sim_loss':1.0, 'tab_total_dist_loss':1.0,
                'vis_recon_loss':1.0, 'tab_recon_loss':1.0,
                'vis_commit_loss':1.0}
    loss_cfg = OmegaConf.create(loss_cfg)

    model = MM_MAE(text_cfg=text_model_cfg, vis_cfg=vis_cfg, tab_cfg=tab_model_cfg, mid_cfg=mid_model_cfg, loss_cfg=loss_cfg)
    model.cuda()


    class CustomDataset(Dataset):
        def __init__(self, tensor_a, tensor_b):
            """
            Args:
                tensor_a (torch.Tensor): Shape [data_num, d1]
                tensor_b (torch.Tensor): Shape [data_num, d2]
            """
            self.tensor_a = tensor_a
            self.tensor_b = tensor_b
            self.data_num = tensor_a.shape[0]

        def __len__(self):
            return self.data_num

        def __getitem__(self, idx):
            num_feat = self.tensor_a[idx]  # Shape [d1]
            cat_feat = self.tensor_b[idx]  # Shape [d2]
            image = torch.randn(1, 240, 480, 480)  # Random tensor for image
            pos_prompt = 'Lung cancer.'
            neg_prompt = 'No lung cancer.'

            return {"num_feat": num_feat, "cat_feat": cat_feat, "image": image,
                    'pos_prompt':pos_prompt, 'neg_prompt':neg_prompt}
        
    dataset = CustomDataset(data_list[0]['train'], data_list[1]['train'])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        if isinstance(batch, dict):
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
        else:
            batch = batch.cuda()
        output = model(batch)
        print('hahaha.')

    print('hahaha')