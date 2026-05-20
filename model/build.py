from model import objectives
from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import Transformer1, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F
from collections import OrderedDict

def l2norm(X, dim=-1, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type

        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
        self.cross_modal_transformer = Transformer1(width=self.embed_dim,
                                                    layers=args.cmt_depth,
                                                    heads=self.embed_dim //
                                                    64)
        scale = self.cross_modal_transformer.width**-0.5
        
        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                        ('gelu', QuickGELU()),
                        ('ln', LayerNorm(self.embed_dim)),
                        ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
        # init mlm head
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)  


    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x, atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)
        return i_tse_f.float()

    def encode_text_tse(self, text):
        x, atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], 
                                                tau=self.args.tau,
                                                margin=self.args.margin,
                                                loss_type=self.loss_type,
                                                logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],
                                                tau=self.args.tau,
                                                margin=self.args.margin,
                                                loss_type=self.loss_type,
                                                logit_scale=self.logit_scale)
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        epoch=batch['epoch']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        label_hat = batch['label_hat'].to(i_feats.device)

        loss1, loss2,sc_loss = objectives.compute_rbs(
            i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'],
            label_hat=label_hat,
            margin=self.args.margin,
            tau=self.args.tau,
            loss_type=self.loss_type,
            logit_scale=self.logit_scale,
            epoch=epoch,
            alpha=getattr(self.args, 'cgr_alpha', 1.2),
            hard_ratio=getattr(self.args, 'cgr_hard_ratio', 0.18),
            hard_ratio_min=getattr(self.args, 'cgr_hard_ratio_min', 0.15),
            lambda_local_max=getattr(self.args, 'cgr_lambda_local_max', 0.5),
            conf_floor=getattr(self.args, 'cgr_conf_floor', 0.55),
            normalize_weights=getattr(self.args, 'cgr_normalize_weights', True),
            sc_loss_weight=getattr(self.args, 'cgr_sc_loss_weight', 0.03),
            cgr_margin_div=getattr(self.args, 'cgr_margin_div', 0.1),
            cgr_margin_train=getattr(self.args, 'cgr_margin_train', 0.1),
            cgr_div_weight=getattr(self.args, 'cgr_div_weight', 0.0),
            cgr_train_weight=getattr(self.args, 'cgr_train_weight', 0.05),
            warm_local=getattr(self.args, 'cgr_warm_local', 5),
            ramp_local=getattr(self.args, 'cgr_ramp_local', 5),
            warm_alpha=getattr(self.args, 'cgr_warm_alpha', 3)
        )

        mlm_ids = batch['mlm_ids']
        mlm_feats = self.base_model.encode_text1(mlm_ids)
        x = self.cross_former(mlm_feats, image_feats, image_feats)
        x = self.mlm_head(x)

        scores = x.float().reshape(-1, self.args.vocab_size)
        mlm_labels = batch['mlm_labels'].reshape(-1)
        ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

        pred = scores.max(1)[1]
        mlm_label_idx = torch.nonzero(mlm_labels)
        acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
        ret.update({'mlm_acc': acc})
        ret.update({'bge_loss': loss1})
        ret.update({'tse_loss': loss2})
        ret.update({'sc_loss': sc_loss})
        return ret

def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    convert_weights(model)
    return model
