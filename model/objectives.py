import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)

def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss

def compute_TRL_per(scores, pid, margin = 0.2, tau=0.02):       
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2


def compute_InfoNCE_per(scores, logit_scale):
    # scores: [B, B] similarity matrix
    logits = logit_scale * scores  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)  # 正确匹配是对角线
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    loss = (loss_i + loss_t) / 2
    return loss

def compute_TAL_per(scores, pid, tau, margin):

    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t() 
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss 

def cgr_anchor_regularizer(S, detach=True, margin=0.0):
    if detach:
        S = S.detach()
    B = S.size(0)
    diag = S.diag()
    a = torch.argmax(diag)  # anchor index
    s_ai = S[a, :]          # I_anchor vs T_j
    s_ia = S[:, a]          # I_j vs T_anchor
    diff = (s_ai - s_ia).pow(2)            # [B]
    l_cgr_div = F.relu(diff.mean() - margin) 
    return l_cgr_div, diff

def cgr_pseudo_neg_regularizer(S, clean_mask, margin=0.0):
    idx = torch.nonzero(clean_mask, as_tuple=False).squeeze(1)
    if idx.numel() <= 1:
        return S.new_tensor(0.0)

    S_clean = S[idx][:, idx]      
    Bc = S_clean.size(0)

    S_mask = S_clean - torch.eye(Bc, device=S.device) * 1e9
    p = torch.argmax(S_mask, dim=0)          
    spi = S_clean[p, torch.arange(Bc, device=S.device)]
    sip = S_clean[torch.arange(Bc, device=S.device), p]
    diff = (spi - sip).pow(2)
    L_CGR_train = F.relu(diff.mean() - margin)
    return L_CGR_train

def _sc_conf_from_feats(a_feats, b_feats, pid, logit_scale=50.0, eps=1e-12):
    B = a_feats.size(0); device = a_feats.device
    a_norm = F.normalize(a_feats, dim=-1)
    b_norm = F.normalize(b_feats, dim=-1)
    sim = logit_scale * (a_norm @ b_norm.t())   # [B,B]

    pid = pid.view(B)
    pid_mat = pid.expand(B, B)
    neg_mask = (pid_mat != pid_mat.t())

    p_a2b = F.softmax(sim.masked_fill(~neg_mask, float('-inf')), dim=1)
    p_b2a = F.softmax(sim.t().masked_fill(~neg_mask.t(), float('-inf')), dim=1)

    kl_1 = F.kl_div((p_a2b + eps).log(), (p_b2a + eps), reduction='batchmean')
    kl_2 = F.kl_div((p_b2a + eps).log(), (p_a2b + eps), reduction='batchmean')
    sc_loss = 0.5 * (kl_1 + kl_2)

    with torch.no_grad():
        K_a = neg_mask.float().sum(dim=1).clamp(min=1.0)
        K_b = neg_mask.t().float().sum(dim=1).clamp(min=1.0)
        H_a = -(p_a2b * (p_a2b + eps).log()).sum(dim=1) / torch.log(K_a + eps)
        H_b = -(p_b2a * (p_b2a + eps).log()).sum(dim=1) / torch.log(K_b + eps)
        H = 0.5 * (H_a + H_b)
        conf = 1.0 - H
    return conf, sc_loss, sim  

def compute_cgr_loss(i_feats, t_feats, pid, label_hat,
                     logit_scale=50.0, eps=1e-12,
                     i_tse_f=None, t_tse_f=None,
                     lambda_global=0.5, lambda_local=0.5,
                     hard_ratio=0.30, alpha=1.2, conf_floor=0.50,
                     cgr_margin_div=0.0, cgr_margin_train=0.0,
                     cgr_div_weight=0.0,    
                     cgr_train_weight=0.05, 
                     epoch=0):
    device = i_feats.device
    B = i_feats.size(0)

    conf_g, sc_g, Sg = _sc_conf_from_feats(i_feats, t_feats, pid, logit_scale, eps)
    if (i_tse_f is not None) and (t_tse_f is not None):
        conf_l, sc_l, Sl = _sc_conf_from_feats(i_tse_f, t_tse_f, pid, logit_scale, eps)
    else:
        conf_l, sc_l, Sl = conf_g.clone(), torch.zeros((), device=device), Sg

    with torch.no_grad():
        cgr_div_g, diff_g = cgr_anchor_regularizer(Sg, detach=True, margin=cgr_margin_div)
        cgr_div_l, diff_l = cgr_anchor_regularizer(Sl, detach=True, margin=cgr_margin_div)

        diff_g_n = (diff_g - diff_g.min()) / (diff_g.max() - diff_g.min() + 1e-6)
        diff_l_n = (diff_l - diff_l.min()) / (diff_l.max() - diff_l.min() + 1e-6)
        cgr_conf = 1.0 - (0.5 * diff_g_n + 0.5 * diff_l_n)

    conf = (lambda_global * conf_g + lambda_local * conf_l).clamp(0, 1)
    conf = 0.5 * conf + 0.5 * cgr_conf         
    conf = conf_floor + (1.0 - conf_floor) * conf

    w = label_hat.float().to(device).clone()
    hard_mask = torch.zeros_like(w, dtype=torch.bool, device=device)
    clean_idx = (w == 1).nonzero(as_tuple=False).squeeze(1)
    if clean_idx.numel() > 0:
        cvals = torch.nan_to_num(conf[clean_idx], nan=1.0)
        k = max(1, int(hard_ratio * cvals.numel()))
        _, rel_idx = torch.topk(cvals, k, largest=False, sorted=True)
        pick = clean_idx[rel_idx]
        hard_mask[pick] = True
        w[hard_mask] = float(alpha)

    L_CGR_train_g = cgr_pseudo_neg_regularizer(Sg, clean_mask=(w > 0.5), margin=cgr_margin_train)
    L_CGR_train_l = cgr_pseudo_neg_regularizer(Sl, clean_mask=(w > 0.5), margin=cgr_margin_train)
    L_CGR_train = 0.5 * (L_CGR_train_g + L_CGR_train_l)
    sc_loss = 0.5 * (sc_g + sc_l)
    return w, sc_loss, (cgr_train_weight * L_CGR_train)

def compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, pid,
                label_hat=None, tau=0.02, margin=0.1,
                loss_type='TAL', logit_scale=50, m=2,
                epoch=0,
                alpha=1.2, hard_ratio=0.18,
                hard_ratio_min=0.15,
                lambda_local_max=0.5,        
                conf_floor=0.55,
                normalize_weights=True,      
                sc_loss_weight=0.03,
                cgr_margin_div=0.1,          
                cgr_margin_train=0.1,
                cgr_div_weight=0.0,
                cgr_train_weight=0.05,
                warm_local=5,
                ramp_local=5,
                warm_alpha=3,
                debug=False):

    if epoch <= warm_local:
        lam_local = 0.0
    else:
        p = min(max((epoch - warm_local) / max(1, ramp_local), 0.0), 1.0)
        lam_local = p * lambda_local_max
    lam_global = 1.0 - lam_local

    alpha_now = 1.0 + (min(epoch, warm_alpha) / max(1, warm_alpha)) * (alpha - 1.0)
    hard_ratio_now = hard_ratio_min + (min(epoch, warm_alpha) / max(1, warm_alpha)) * (hard_ratio - hard_ratio_min)

    w_raw, sc_loss, cgr_train = compute_cgr_loss(
        i_feats, t_feats, pid, label_hat,
        logit_scale=logit_scale,
        i_tse_f=i_tse_f, t_tse_f=t_tse_f,
        lambda_global=lam_global, lambda_local=lam_local,
        hard_ratio=hard_ratio_now, alpha=alpha_now, conf_floor=conf_floor,
        cgr_margin_div=cgr_margin_div, cgr_margin_train=cgr_margin_train,
        cgr_div_weight=cgr_div_weight, cgr_train_weight=cgr_train_weight,
        epoch=epoch
    )

    loss_bgm_vec, _ = compute_per_loss(i_feats,   t_feats,   pid, tau, margin, loss_type, logit_scale)
    loss_tse_vec, _ = compute_per_loss(i_tse_f,   t_tse_f,   pid, tau, margin, loss_type, logit_scale)

    if normalize_weights:
        w = w_raw / w_raw.mean().clamp(min=1e-6)
    else:
        w = w_raw

    if loss_type in ['TAL', 'TRL']:
        loss_bgm = (w * loss_bgm_vec).sum()
        loss_tse = (w * loss_tse_vec).sum()
    else:
        denom = w.sum().clamp(min=1.0)
        loss_bgm = (w * loss_bgm_vec).sum() / denom
        loss_tse = (w * loss_tse_vec).sum() / denom

    reg_loss = sc_loss_weight * sc_loss + cgr_train

    # if debug:
    #     clean = (label_hat > 0.5)
    #     hard  = (w_raw > 1.0) & clean
    #     noisy = ~clean
    #     print(f"[rbs] e={epoch} "
    #           f"lam_global={lam_global:.2f} lam_local={lam_local:.2f} "
    #           f"alpha_now={alpha_now:.2f} hard_ratio_now={hard_ratio_now:.2f} | "
    #           f"clean={clean.sum().item()} hard={hard.sum().item()} noisy={noisy.sum().item()} | "
    #           f"w_raw(min/mean/max)={w_raw.min():.3f}/{w_raw.mean():.3f}/{w_raw.max():.3f} "
    #           f"w(min/mean/max)={w.min():.3f}/{w.mean():.3f}/{w.max():.3f} | "
    #           f"sc={sc_loss:.4f} cgr_train={cgr_train:.4f}")

    return loss_bgm, loss_tse, reg_loss



def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    else:
        exit()

    return per_loss, scores.diag()





