import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def get_all_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                cond_BD: Optional[torch.Tensor],
                prefix_len: Optional[int] = None):  # 添加prefix_len参数
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
            
        # 获取所有位置的logits
        all_logits = self.head(self.head_nm(h.float(), cond_BD).float()).float()
        
        # 如果指定了prefix_len,则返回prefix部分的logits
        if prefix_len is not None:
            return all_logits[:, :prefix_len], all_logits[:, prefix_len:]
        return all_logits
        
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )

class SDVAR(nn.Module):
    def __init__(
        self,
        draft_model,
        target_model,
        similarity_thresh: float = 0.8
    ):

        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.similarity_thresh = similarity_thresh
    
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        gamma: int = 10
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        这里可以考虑top_k, top_p是否需要将target_model和draft_model分开，这样可以更加有效？
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param gamma: draft_model 每次向前预测的数量
        :param warmup_steps: 参照code表示最开始目标模型预测的数量
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        
        ###### 初始化参数

        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1
        # rng, label_B可以公用，不涉及
        if g_seed is not None:
            self.draft_model.rng.manual_seed(g_seed)
            self.target_model.rng.manual_seed(g_seed)
            rng = self.draft_model.rng
        else:
            rng = None
        if label_B is None:
            label_B = torch.multinomial(
                self.draft_model.uniform_prob, num_samples=B, replacement=True, generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.draft_model.num_classes if label_B < 0 else label_B,
                device=self.draft_model.lvl_1L.device
            )
        ###### 不可共用的内容，embed形状不一样所以没有办法共用
        draft_sos = draft_cond_BD = self.draft_model.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.draft_model.num_classes)), dim=0)
        )   # shape: (2B, C)
        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)
        draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC
        draft_first_token_map = (
            draft_sos.unsqueeze(1).expand(2*B, self.draft_model.first_l, -1)
            + self.draft_model.pos_start.expand(2*B, self.draft_model.first_l, -1)
            + draft_lvl_pos[:, :self.draft_model.first_l]
        )
        # 这个其实没什么用，因为其实draft的f_hat不会用到
        draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae, self.draft_model.patch_nums[-1], self.draft_model.patch_nums[-1])

        target_sos = target_cond_BD = self.target_model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.target_model.num_classes)), dim=0))
        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)
        target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC
        target_first_token_map = target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1) \
            + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1) \
            + target_lvl_pos[:, :self.target_model.first_l]      
        target_f_hat = target_sos.new_zeros(B, self.target_model.Cvae, self.target_model.patch_nums[-1], self.target_model.patch_nums[-1])


        ###### 推测解码用的参数
        accept_si = 0
        accept_L = 0
        accept_token_hub = []
        self.exit_points = [0] * len(self.patch_nums)
        self.exit_points[0] = self.patch_nums[0] ** 2
        for i in range(1, len(self.patch_nums)):
            self.exit_points[i] = self.exit_points[i - 1] + self.patch_nums[i] ** 2


        ###### KVCache
        for b in self.draft_model.blocks:
            b.attn.kv_caching(True)
        for b in self.target_model.blocks:
            b.attn.kv_caching(True)

        ###### 循环：生成-验证
        while accept_si < self.num_stages_minus_1:
            local_si = accept_si
            local_L = accept_L
            draft_steps = min(gamma, self.num_stages_minus_1 -si)
            # backup_xxx
            
            # draft_model生成draft
            for si, pn in enumerate(self.patch_nums):
                # 一直跳到已接受的
                if si < accept_si:
                    continue
                # 生成到已经接受的
                elif si == accept_si:
                    draft_token_hub = accept_token_hub
                    draft_next_token_map = draft_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
                    draft_next_token_map = torch.cat([draft_first_token_map,draft_next_token_map],dim=1)
                
                AdaLNSelfAttn.forward
                
                
                # 生成足够的多的draft以后退出
                if si == accept_si + draft_steps - 1:
                    break
            

            
            # target_model验证draft
            for si, pn in enumerate(self.patch_nums):
                # 没有到达目标层之前直接跳过
                if si < accept_si + draft_steps - 1:
                    continue
                # 到达目标层后接受token
                if si == accept_si + draft_steps - 1:
                    target_L = local_L
                    target_token_hub = draft_token_hub
                    target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:,1:pindex] 
                # 接受目标层后使用block_wise的掩码生成下一层
                d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                lvl_1L = dT[:, 0].contiguous()
                self.register_buffer('lvl_1L', lvl_1L)
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
                self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
                for b in self.target_model.blocks:
                    x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias_for_masking)
                
                # 逐层判断是否接受
                for si in range(local_si, local_si + draft_steps - 1):
                    pn = self.patch_nums[si]
                    # 接受则将补齐
                    if measure_similarity_with_target_parallel()==True:
                        accept_si = si
                        accept_L += pn * pn
                        accept_token_hub.append()
                        pass
                    # 拒绝则退出并回退
                    else:
                        # 小模型回退
                        # 大模型回退
                        break
                # 完成后退出
                break

        ###### 结束并返回结果
        for b in self.draft_model.blocks:
            b.attn.kv_caching(False)
        for b in self.target_model.blocks:
            b.attn.kv_caching(False)
        return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)
    
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_test(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        entry_num: int = 10
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        这里可以考虑top_k, top_p是否需要将target_model和draft_model分开，这样可以更加有效？
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param gamma: draft_model 每次向前预测的数量
        :param warmup_steps: 参照code表示最开始目标模型预测的数量
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        
        ###### 初始化参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1

        total_stages = len(self.patch_nums)

        if g_seed is not None:
            self.draft_model.rng.manual_seed(g_seed)
        else:
            self.draft_model.rng = None

        draft_label_B = label_B
        if draft_label_B is None:
            draft_label_B = torch.multinomial(
                self.draft_model.uniform_prob, num_samples=B, replacement=True, generator=self.draft_model.rng
            ).reshape(B)
        elif isinstance(draft_label_B, int):
            draft_label_B = torch.full(
                (B,),
                fill_value=self.draft_model.num_classes if draft_label_B < 0 else draft_label_B,
                device=self.draft_model.lvl_1L.device
            )

        draft_sos = draft_cond_BD = self.draft_model.class_emb(
            torch.cat((draft_label_B, torch.full_like(draft_label_B, fill_value=self.draft_model.num_classes)), dim=0)
        )

        draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC
        draft_next_token_map = (
            draft_sos.unsqueeze(1).expand(2*B, self.draft_model.first_l, -1)
            + self.draft_model.pos_start.expand(2*B, self.draft_model.first_l, -1)
            + draft_lvl_pos[:, :self.draft_model.first_l]
        )
        
        draft_cur_L = 0
        draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae,
                                          self.draft_model.patch_nums[-1],
                                          self.draft_model.patch_nums[-1])
        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)
        draft_token_hub =[]
        
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            
            if si >= entry_num:
                break

            ratio = si / self.num_stages_minus_1
            draft_cur_L += pn*pn
            x = draft_next_token_map
            
            AdaLNSelfAttn.forward
            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)            
            
            t = cfg * ratio
            draft_logits_BlV = (1+t)*draft_logits_BlV[:B] - t*draft_logits_BlV[B:]  # (B, l, V)

            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV,
                rng=self.draft_model.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            if not more_smooth:
                draft_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), tau=draft_gum_t, hard=False, dim=-1, rng=self.draft_model.rng
                    ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            draft_h_BChw = draft_h_BChw.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, draft_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, draft_h_BChw
            )

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1,2)
                draft_token_hub.append(draft_next_token_map)
                draft_next_token_map = (
                    self.draft_model.word_embed(draft_next_token_map)
                    + draft_lvl_pos[:, draft_cur_L : draft_cur_L + next_pn*next_pn]
                )
                draft_next_token_map = draft_next_token_map.repeat(2,1,1)

            if si == self.num_stages_minus_1:
                for blk in self.draft_model.blocks:
                    blk.attn.kv_caching(False)
                return self.draft_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

        # draft模型生成完毕  
        draf_token_hub = torch.cat(draft_token_hub, dim = 1)      
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        
        # return self.draft_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
        ###### target模型接受draft模型生成的内容然后生成最后一层的内容
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        pindex = exit_points[entry_num]

        if g_seed is not None:
            self.target_model.rng.manual_seed(g_seed)
        else:
            self.target_model.rng = None

        target_label_B = label_B
        if target_label_B is None:
            target_label_B = torch.multinomial(
                self.target_model.uniform_prob, num_samples=B, replacement=True, generator=self.target_model.rng
            ).reshape(B)
        elif isinstance(target_label_B, int):
            target_label_B = torch.full(
                (B,),
                fill_value=self.target_model.num_classes if target_label_B < 0 else target_label_B,
                device=self.target_model.lvl_1L.device
            )

        target_sos = target_cond_BD = self.target_model.class_emb(
            torch.cat((target_label_B, torch.full_like(target_label_B, fill_value=self.target_model.num_classes)), dim=0)
        )
        target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC
        target_token_hub = draft_token_hub
        
        # target_next_token_map = target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1) \
        #     + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1) \
        #     + target_lvl_pos[:, :self.target_model.first_l]
        
        target_first_token_map = target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1) \
            + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1) \
            + target_lvl_pos[:, :self.target_model.first_l]
        
        target_cur_L = 0
        target_f_hat = target_sos.new_zeros(B, self.target_model.Cvae,
                                            self.target_model.patch_nums[-1],
                                            self.target_model.patch_nums[-1])

        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)
        

        # 接受之前生成的做为target_model输出的prefix
        target_next_token_map = target_token_hub
        target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:,1:pindex]  
        target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        target_next_token_map = torch.cat([target_first_token_map,target_next_token_map],dim=1)
        print(self.target_model.word_embed(target_token_hub).shape,flush=True)
        print(target_lvl_pos[:, 1:pindex].shape,flush=True)
        attn_bias = self.target_model.attn_bias_for_masking[:,:,0:pindex,0:pindex]
        
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            target_cur_L += pn*pn
            t = cfg * ratio 
            
            if si < entry_num:
                continue

            x = target_next_token_map
            AdaLNSelfAttn.forward
            # if si == entry_num:
                # print("attention bias shape:",attn_bias.shape, flush=True)
                # print("cond_BD_or_gss.shape:",cond_BD_or_gss.shape, flush=True)
                # print("x.shape:",x.shape, flush=True)
            #    for b in self.target_model.blocks:
            #        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
            if si >= entry_num:
                for b in self.target_model.blocks:
                    x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
            # for b in self.target_model.blocks:
            #     x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
            target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)


            if si == entry_num:
                target_logits_BlV[:B,target_cur_L-pn*pn:target_cur_L] = (1+t) * target_logits_BlV[:B,target_cur_L-pn*pn:target_cur_L] - t * target_logits_BlV[B:,target_cur_L-pn*pn:target_cur_L]

                new_L = 0
                for a, b in enumerate(self.patch_nums[0:entry_num+1]):
                    target_idx_Bl=sample_with_top_k_top_p_(
                        target_logits_BlV[:B,new_L:new_L + self.patch_nums[a] ** 2], 
                        rng=self.target_model.rng, 
                        top_k=top_k, 
                        top_p=top_p, 
                        num_samples=1
                    )[:, :, 0]
                    new_L += b*b
                
                target_logits_BlV = target_logits_BlV[:B,target_cur_L-pn*pn:target_cur_L]

            # target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
            # target_idx_Bl = sample_with_top_k_top_p_(target_logits_BlV, rng=self.target_model.rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            
            elif si > entry_num:
                target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
                target_idx_Bl = sample_with_top_k_top_p_(
                    target_logits_BlV,
                    rng=self.target_model.rng,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1
                )[:, :, 0]


            if not more_smooth: # this is the default case
                target_h_BChw = self.target_model.vae_quant_proxy[0].embedding(target_idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio),
                    tau=target_gum_t,
                    hard=False, dim=-1,
                    rng=self.target_model.rng
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose_(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), target_f_hat, target_h_BChw
            )
            
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            
        # target模型生成完成
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)   
                    
        return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_warmup(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        gamma: int = 10,
        warmup_steps: int = 5  # 最开始由 target_model 生成的层数,效果类似entry_num
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        这里可以考虑 top_k, top_p 是否需要将 target_model 和 draft_model 分开设置
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: 是否使用 gumbel softmax 平滑预测（仅用于可视化，不用于 FID/IS 评价）
        :param gamma: draft_model 每次向前预测的层数
        :param warmup_steps: 最开始 target_model 预测的层数
        :return: 最终生成的图像
        """
        ######################################
        # 初始化参数：patch_nums、随机种子、标签等
        ######################################
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1

        if g_seed is not None:
            self.draft_model.rng.manual_seed(g_seed)
            self.target_model.rng.manual_seed(g_seed)
            rng = self.draft_model.rng
        else:
            rng = None

        if label_B is None:
            label_B = torch.multinomial(
                self.draft_model.uniform_prob, num_samples=B, replacement=True, generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.draft_model.num_classes if label_B < 0 else label_B,
                device=self.draft_model.lvl_1L.device
            )

        ######################################
        # 初始化 draft_model 的输入（非共享部分）
        ######################################
        draft_sos = draft_cond_BD = self.draft_model.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.draft_model.num_classes)), dim=0)
        )  # shape: (2B, C)
        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)
        draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC
        draft_first_token_map = (
            draft_sos.unsqueeze(1).expand(2 * B, self.draft_model.first_l, -1)
            + self.draft_model.pos_start.expand(2 * B, self.draft_model.first_l, -1)
            + draft_lvl_pos[:, :self.draft_model.first_l]
        )
        # 草稿生成过程中的 f_hat（最终不用于生成图像，仅占位）
        draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae,
                                        self.draft_model.patch_nums[-1],
                                        self.draft_model.patch_nums[-1])

        ######################################
        # 初始化 target_model 的输入
        ######################################
        target_sos = target_cond_BD = self.target_model.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.target_model.num_classes)), dim=0)
        )
        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)
        target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC
        target_first_token_map = (
            target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1)
            + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1)
            + target_lvl_pos[:, :self.target_model.first_l]
        )
        target_f_hat = target_sos.new_zeros(B, self.target_model.Cvae,
                                            self.target_model.patch_nums[-1],
                                            self.target_model.patch_nums[-1])

        ######################################
        # 计算每一阶段结束时 token 的累积数量（exit_points）
        ######################################
        self.exit_points = [0] * len(self.patch_nums)
        self.exit_points[0] = self.patch_nums[0] ** 2
        for i in range(1, len(self.patch_nums)):
            self.exit_points[i] = self.exit_points[i - 1] + self.patch_nums[i] ** 2

        ######################################
        # Warmup 阶段：先用 target_model 生成 warmup_steps 层
        ######################################
        target_token_hub = []
        target_next_token_map = target_first_token_map  # 初始输入
        current_L = 0
        for si, pn in enumerate(self.patch_nums):
            if si >= warmup_steps:
                break
            ratio = si / self.num_stages_minus_1
            current_L += pn * pn
            x = target_next_token_map
            # 对 target_model 的每个 block 做前向传播（假设内部已调用 AdaLNSelfAttn.forward）
            for blk in self.target_model.blocks:
                x = blk(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
            logits = self.target_model.get_logits(x, target_cond_BD)
            t = cfg * ratio
            # 对 logits 进行 CFG 调整（注意取前 B 部分与后 B 部分的区分）
            # cfg调整如何调整到外侧？值得考虑的点
            logits = (1 + t) * logits[:B, current_L - pn * pn:current_L] - t * logits[B:, current_L - pn * pn:current_L]
            idx = sample_with_top_k_top_p_(logits, rng=self.target_model.rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:
                h_BChw = self.target_model.vae_quant_proxy[0].embedding(idx)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(
                    logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=self.target_model.rng
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            h_BChw = h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, pn, pn)
            target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), target_f_hat, h_BChw
            )
            # 收集每阶段生成的 token（供后续投影到 draft_model 时使用）
            token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
            target_token_hub.append(token_map)
            # POSSIBLE BUG 这里是可能的 Bug，一般是总体的num_minus_1这里是warmup_step-1
            if si < warmup_steps - 1:
                next_pn = self.patch_nums[si + 1]
                target_next_token_map = (
                    self.target_model.word_embed(token_map)
                    + target_lvl_pos[:, current_L: current_L + next_pn * next_pn]
                )
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)
        # 合并 warmup 阶段 token
        target_token_hub = torch.cat(target_token_hub, dim=1)

        # 设置草稿阶段已接受的层数及 token（初始阶段为 warmup_steps 生成的部分）
        accept_si = warmup_steps
        accept_L = current_L
        accept_token_hub = target_token_hub

        ######################################
        # 开启 KV cache
        ######################################
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        ######################################
        # 循环：草稿生成（draft）—验证（target_model）
        ######################################
        while accept_si < self.num_stages_minus_1:
            local_si = accept_si
            local_L = accept_L
            # 本轮草稿预测步数，不能超过剩余的层数
            draft_steps = min(gamma, self.num_stages_minus_1 - accept_si)

            ###### 备份当前状态（如果需要回退，可保存 target_f_hat、target_next_token_map 等） ######
            # backup_target_f_hat = target_f_hat.clone()  # 示例

            ######################################
            # Step 1: 使用 draft_model 从当前层开始预测 draft_steps 层
            ######################################
            # 初始化 draft_next_token_map。如果处于 warmup 后首次进入，则用 target 的 token hub 投影到 draft 空间
            if accept_si == warmup_steps:
                # 注意：这里简单使用 word_embed 投影，实际可能需要专门的映射模块
                draft_prefix = self.draft_model.word_embed(accept_token_hub)
                draft_next_token_map = torch.cat([draft_first_token_map, draft_prefix], dim=1)
            # 对于 CFG，扩充 batch size
            draft_next_token_map = draft_next_token_map.repeat(2, 1, 1)

            current_draft_token_hub = []
            for si, pn in enumerate(self.patch_nums):
                if si < accept_si:
                    continue
                # 当达到本轮最后一层时退出草稿生成
                if si >= accept_si and si < accept_si + draft_steps:
                    ratio = si / self.num_stages_minus_1
                    x = draft_next_token_map
                    # 如有需要，可为第一层设置注意力 mask
                    attn_bias = None
                    if si == accept_si:
                        attn_bias = self.draft_model.attn_bias_for_masking[:, :, :self.exit_points[si], :self.exit_points[si]]
                    for blk in self.draft_model.blocks:
                        x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=attn_bias)
                    logits = self.draft_model.get_logits(x, draft_cond_BD)
                    t = cfg * ratio
                    logits = (1 + t) * logits[:B] - t * logits[B:]
                    idx = sample_with_top_k_top_p_(logits, rng=self.draft_model.rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                    if not more_smooth:
                        h_BChw = self.draft_model.vae_quant_proxy[0].embedding(idx)
                    else:
                        gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                        h_BChw = gumbel_softmax_with_rng(
                            logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=self.draft_model.rng
                        ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                    h_BChw = h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)
                    draft_f_hat, draft_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                        si, len(self.patch_nums), draft_f_hat, h_BChw
                    )
                    token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1, 2)
                    current_draft_token_hub.append(token_map)
                    if si == accept_si + draft_steps - 1:
                        break
            if current_draft_token_hub:
                draft_token_hub = torch.cat(current_draft_token_hub, dim=1)
            else:
                draft_token_hub = None

            ######################################
            # Step 2: 使用 target_model 验证草稿结果
            ######################################
            # 将 draft_model 输出投影到 target_model 空间
            projected_tokens = self.target_model.word_embed(draft_token_hub)
            # 依据 exit_points 得到当前草稿层结束时的位置索引
            pindex = self.exit_points[accept_si + draft_steps - 1]
            # 构造 target_model 后续输入（拼接初始 token 与草稿预测并加上位置编码）
            target_next_token_map = torch.cat(
                [target_first_token_map, projected_tokens + target_lvl_pos[:, 1:pindex]], dim=1
            )

            # 构造 block-wise attention mask（简化版）
            L = target_next_token_map.shape[1]
            d = torch.cat([torch.full((pn * pn,), i, device=target_next_token_map.device)
                        for i, pn in enumerate(self.patch_nums)]).view(1, L, 1)
            dT = d.transpose(1, 2)
            attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L)
            for blk in self.target_model.blocks:
                target_next_token_map = blk(x=target_next_token_map,
                                            cond_BD=target_cond_BD_or_gss,
                                            attn_bias=attn_bias_for_masking)

            # 调用验证函数，测量草稿与 target_model 预测的相似性
            # 此处 measure_similarity_with_target_parallel 为占位函数，返回 True 表示接受当前草稿
            if True:
                # 接受当前草稿，更新 accept_si 与累计 token 数
                accepted_stage = accept_si + draft_steps - 1
                pn = self.patch_nums[accepted_stage]
                accept_si = accepted_stage + 1
                accept_L += pn * pn
                # 将验证通过的 token 加入已接受的 token hub
                if accept_token_hub is None:
                    accept_token_hub = draft_token_hub
                else:
                    accept_token_hub = torch.cat([accept_token_hub, draft_token_hub], dim=1)
                # target_f_hat 也可以在这里更新（此处省略细节）
            else:
                # 如果验证未通过，则进行回退处理（此处仅作提示，具体策略需根据业务需求设计）
                print("Verification failed at stage:", accept_si + draft_steps - 1)
                # 可选择回退至 backup 状态或重新生成当前草稿
                break

        ######################################
        # 关闭 KV cache
        ######################################
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)

        ######################################
        # 最终利用 target_model 的 f_hat 生成图像
        ######################################
        final_img = self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)
        return final_img

    torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_tdt(
            self,
            B: int,
            label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None,
            cfg: float = 1.5,
            top_k: int = 0,
            top_p: float = 0.0,
            more_smooth: bool = False,
            entry_num_1: int = 5,   # 第1阶段（target_model）的层数
            entry_num_2: int = 10   # 第2阶段（draft_model）结束后转换到最终target_model的层数
        ) -> torch.Tensor:
        """
        仅用于自回归推理。新的生成流程为：
        stage1: target_model 生成初始部分（层 0 ~ entry_num_1-1）
        stage2: draft_model 接收 stage1 的输出继续生成（层 entry_num_1 ~ entry_num_2-1）
        stage3: target_model 最终生成剩余部分（层 entry_num_2 ~ end）
        原有的 entry_num 被拆分为 entry_num_1 和 entry_num_2 表示两处模型转换。
        """
        ###### 初始化参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1
        total_stages = len(self.patch_nums)
        exit_points = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]

        #############################################################
        ###### Stage 1: target_model 生成初始部分 (0 -> entry_num_1)
        if g_seed is not None:
            self.target_model.rng.manual_seed(g_seed)
        else:
            self.target_model.rng = None

        target_label_B = label_B
        if target_label_B is None:
            target_label_B = torch.multinomial(
                self.target_model.uniform_prob, num_samples=B, replacement=True, generator=self.target_model.rng
            ).reshape(B)
        elif isinstance(target_label_B, int):
            target_label_B = torch.full(
                (B,),
                fill_value=self.target_model.num_classes if target_label_B < 0 else target_label_B,
                device=self.target_model.lvl_1L.device
            )

        target_sos = target_cond_BD = self.target_model.class_emb(
            torch.cat((target_label_B, torch.full_like(target_label_B, fill_value=self.target_model.num_classes)), dim=0)
        )

        target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC
        target_first_token_map = (
            target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1)
            + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1)
            + target_lvl_pos[:, :self.target_model.first_l]
        )

        target_next_token_map = target_first_token_map

        target_cur_L = 0
        target_f_hat = target_sos.new_zeros(B, self.target_model.Cvae,
                                            self.target_model.patch_nums[-1],
                                            self.target_model.patch_nums[-1])
        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)
        target_token_hub_stage1 = []

        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):

            if si >= entry_num_1:
                break

            ratio = si / self.num_stages_minus_1
            target_cur_L += pn * pn
            x = target_next_token_map

            AdaLNSelfAttn.forward
            for blk in self.target_model.blocks:
                x = blk(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
            target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)

            t = cfg * ratio
            target_logits_BlV = (1 + t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
            
            target_idx_Bl = sample_with_top_k_top_p_(
                target_logits_BlV,
                rng=self.target_model.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            if not more_smooth:
                target_h_BChw = self.target_model.vae_quant_proxy[0].embedding(target_idx_Bl)
            else:
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio), tau=target_gum_t, hard=False, dim=-1, rng=self.target_model.rng
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, target_f_hat, target_h_BChw
            )

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si + 1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_token_hub_stage1.append(target_next_token_map)
                target_next_token_map = (
                    self.target_model.word_embed(target_next_token_map)
                    + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                )
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)

            if si == self.num_stages_minus_1:
                for blk in self.target_model.blocks:
                    blk.attn.kv_caching(False)
                return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)
                
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)
        target_token_hub_stage1 = torch.cat(target_token_hub_stage1, dim=1)

        #############################################################
        ###### Stage 2: draft_model 生成中间部分 (entry_num_1 -> entry_num_2)

        pindex1 = exit_points[entry_num_1]

        if g_seed is not None:
            self.draft_model.rng.manual_seed(g_seed)
        else:
            self.draft_model.rng = None

        draft_label_B = label_B
        if draft_label_B is None:
            draft_label_B = torch.multinomial(
                self.draft_model.uniform_prob, num_samples=B, replacement=True, generator=self.draft_model.rng
            ).reshape(B)
        elif isinstance(draft_label_B, int):
            draft_label_B = torch.full(
                (B,),
                fill_value=self.draft_model.num_classes if draft_label_B < 0 else draft_label_B,
                device=self.draft_model.lvl_1L.device
            )

        draft_sos = draft_cond_BD = self.draft_model.class_emb(
            torch.cat((draft_label_B, torch.full_like(draft_label_B, fill_value=self.draft_model.num_classes)), dim=0)
        )
        draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC
        # 用 Stage1 输出初始化 draft 模型的输入（类似于原先 draft 模型的 sos）
        
        draft_first_token_map = draft_sos.unsqueeze(1).expand(2 * B, self.draft_model.first_l, -1) \
            + self.draft_model.pos_start.expand(2 * B, self.draft_model.first_l, -1) \
            + draft_lvl_pos[:, :self.draft_model.first_l]
        
        draft_cur_L = 0
        draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae,
                                        self.draft_model.patch_nums[-1],
                                        self.draft_model.patch_nums[-1])
        
        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)
        
        # 将 target_model 生成的 prefix 经 word_embed 转换到 draft_model 的空间
        draft_prefix = self.draft_model.word_embed(target_token_hub_stage1) \
                    + draft_lvl_pos[:, 1:pindex1]
        draft_prefix = draft_prefix.repeat(2,1,1)
        draft_next_token_map = torch.cat([draft_first_token_map, draft_prefix], dim=1)


        draft_attn_bias = self.draft_model.attn_bias_for_masking[:,:,0:pindex1,0:pindex1]

        draft_token_hub = []
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            draft_cur_L += pn * pn
            t = cfg * ratio
            
            if si < entry_num_1:
                continue
            if si >= entry_num_2:
                break

            x = draft_next_token_map
            AdaLNSelfAttn.forward
            if si == entry_num_1:
                for blk in self.draft_model.blocks:
                    x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=draft_attn_bias)
            elif si > entry_num_1:
                for blk in self.draft_model.blocks:
                    x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)

            if si == entry_num_1:
                draft_logits_BlV[:B,draft_cur_L-pn*pn:draft_cur_L] = (1+t) * draft_logits_BlV[:B,draft_cur_L-pn*pn:draft_cur_L] - t * draft_logits_BlV[B:,draft_cur_L-pn*pn:draft_cur_L]

                new_L = 0
                for a, b in enumerate(self.patch_nums[0:entry_num_1+1]):
                    draft_idx_Bl=sample_with_top_k_top_p_(
                        draft_logits_BlV[:B,new_L:new_L + self.patch_nums[a] ** 2], 
                        rng=self.draft_model.rng, 
                        top_k=top_k, 
                        top_p=top_p, 
                        num_samples=1
                    )[:, :, 0]
                    new_L += b*b
                
                draft_logits_BlV = draft_logits_BlV[:B,draft_cur_L-pn*pn:draft_cur_L]

            elif si > entry_num_1:
                draft_logits_BlV = (1 + t) * draft_logits_BlV[:B] - t * draft_logits_BlV[B:]
                draft_idx_Bl = sample_with_top_k_top_p_(
                    draft_logits_BlV, 
                    rng=self.draft_model.rng, 
                    top_k=top_k, 
                    top_p=top_p, 
                    num_samples=1
                )[:, :, 0]

            if not more_smooth:
                draft_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio),
                    tau=draft_gum_t,
                    hard=False, dim=-1,
                    rng=self.draft_model.rng
                ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            draft_h_BChw = draft_h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, draft_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, draft_h_BChw
            )

            if si != self.num_stages_minus_1:
                next_pn = self.patch_nums[si + 1]
                draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1, 2)
                draft_token_hub.append(draft_next_token_map)
                draft_next_token_map = (
                    self.draft_model.word_embed(draft_next_token_map)
                    + draft_lvl_pos[:, draft_cur_L: draft_cur_L + next_pn * next_pn]
                )
                draft_next_token_map = draft_next_token_map.repeat(2, 1, 1)
            
            if si == self.num_stages_minus_1:
                for blk in self.draft_model.blocks:
                    blk.attn.kv_caching(False)
                return self.draft_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
        
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        draft_token_hub = torch.cat(draft_token_hub, dim=1)
        print(draft_token_hub.shape,flush=True)

        #############################################################
        ###### Stage 3: target_model 最终生成 (entry_num_2 -> end)

        pindex2 = exit_points[entry_num_2]
        print(pindex2,flush=True)

        if g_seed is not None:
            self.target_model.rng.manual_seed(g_seed)
        else:
            self.target_model.rng = None

        target_label_B = label_B
        if target_label_B is None:
            target_label_B = torch.multinomial(
                self.target_model.uniform_prob, num_samples=B, replacement=True, generator=self.target_model.rng
            ).reshape(B)
        elif isinstance(target_label_B, int):
            target_label_B = torch.full(
                (B,),
                fill_value=self.target_model.num_classes if target_label_B < 0 else target_label_B,
                device=self.target_model.lvl_1L.device
            )

        target_sos = target_cond_BD = self.target_model.class_emb(
            torch.cat((target_label_B, torch.full_like(target_label_B, fill_value=self.target_model.num_classes)), dim=0)
        )
        # target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC
        target_token_hub = torch.cat([target_token_hub_stage1,draft_token_hub], dim = 1)
        
        target_first_token_map = target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1) \
            + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1) \
            + target_lvl_pos[:, :self.target_model.first_l]

        target_cur_L = 0
        target_f_hat = target_sos.new_zeros(B, self.target_model.Cvae,
                                            self.target_model.patch_nums[-1],
                                            self.target_model.patch_nums[-1])
        
        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)

        # 使用 draft_model 的输出作为 prefix
        print(self.target_model.word_embed(target_token_hub).shape,flush=True)
        print(target_lvl_pos[:, 1:pindex2].shape,flush=True)
        target_next_token_map = self.target_model.word_embed(target_token_hub) + target_lvl_pos[:, 1:pindex2]
        target_next_token_map = target_next_token_map.repeat(2, 1, 1)
        target_next_token_map = torch.cat([target_first_token_map, target_next_token_map], dim=1)

        attn_bias = self.target_model.attn_bias_for_masking[:, :, 0:pindex2, 0:pindex2]
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):
            if si < entry_num_2:
                continue
            ratio = si / self.num_stages_minus_1
            target_cur_L += pn * pn
            t = cfg * ratio
            x = target_next_token_map
            if si == entry_num_2:
                for b in self.target_model.blocks:
                    x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                target_logits_BlV[:B, target_cur_L - pn * pn: target_cur_L] = (
                    (1 + t) * target_logits_BlV[:B, target_cur_L - pn * pn: target_cur_L]
                    - t * target_logits_BlV[B:, target_cur_L - pn * pn: target_cur_L]
                )
                new_L = 0
                for a, b in enumerate(self.patch_nums[0:entry_num_2 + 1]):
                    target_idx_Bl = sample_with_top_k_top_p_(
                        target_logits_BlV[:B, new_L:new_L + self.patch_nums[a] ** 2],
                        rng=self.target_model.rng,
                        top_k=top_k,
                        top_p=top_p,
                        num_samples=1
                    )[:, :, 0]
                    new_L += b * b
                target_logits_BlV = target_logits_BlV[:B, target_cur_L - pn * pn: target_cur_L]
            elif si > entry_num_2:
                for b in self.target_model.blocks:
                    x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                target_logits_BlV = (1 + t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
                target_idx_Bl = sample_with_top_k_top_p_(
                    target_logits_BlV, rng=self.target_model.rng, top_k=top_k, top_p=top_p, num_samples=1
                )[:, :, 0]
            if not more_smooth:
                target_h_BChw = self.target_model.vae_quant_proxy[0].embedding(target_idx_Bl)
            else:
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio), tau=target_gum_t, hard=False, dim=-1, rng=self.target_model.rng
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            target_h_BChw = target_h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, pn, pn)
            target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), target_f_hat, target_h_BChw
            )
            if si != self.num_stages_minus_1:
                next_pn = self.patch_nums[si + 1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_next_token_map = (
                    self.target_model.word_embed(target_next_token_map)
                    + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                )
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)

        return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)

    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_test1(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        entry_num: int = 10, 
        sd_mask: int = 0
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        这里可以考虑top_k, top_p是否需要将target_model和draft_model分开，这样可以更加有效？
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param entry_num: 转换模型的节点
        :param sd_mask: 是否使用我们自己写的block_wise的掩码
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        
        ###### 初始化参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1

        total_stages = len(self.patch_nums)

        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        block_sizes = [p ** 2 for p in patch_nums]
        total_tokens = sum(block_sizes)

        block_ids = []
        for block, size in enumerate(block_sizes):
            block_ids += [block] * size
        block_ids = torch.tensor(block_ids)

        attn_bias_for_sdmasking = torch.full((total_tokens, total_tokens), float('-inf'))

        for i in range(total_tokens):
            for j in range(total_tokens):
                if j > i:
                    continue
                if block_ids[i] == block_ids[j] and i != j:
                    continue
                attn_bias_for_sdmasking[i, j] = 0.0

        attn_bias_for_sdmasking = attn_bias_for_sdmasking.reshape(1, 1, total_tokens, total_tokens)

        if g_seed is not None:
            self.draft_model.rng.manual_seed(g_seed)
        else:
            self.draft_model.rng = None

        draft_label_B = label_B
        if draft_label_B is None:
            draft_label_B = torch.multinomial(
                self.draft_model.uniform_prob, num_samples=B, replacement=True, generator=self.draft_model.rng
            ).reshape(B)
        elif isinstance(draft_label_B, int):
            draft_label_B = torch.full(
                (B,),
                fill_value=self.draft_model.num_classes if draft_label_B < 0 else draft_label_B,
                device=self.draft_model.lvl_1L.device
            )

        draft_sos = draft_cond_BD = self.draft_model.class_emb(
            torch.cat((draft_label_B, torch.full_like(draft_label_B, fill_value=self.draft_model.num_classes)), dim=0)
        )   

        draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC

        draft_first_token_map = (
            draft_sos.unsqueeze(1).expand(2*B, self.draft_model.first_l, -1)
            + self.draft_model.pos_start.expand(2*B, self.draft_model.first_l, -1)
            + draft_lvl_pos[:, :self.draft_model.first_l]
        )

        draft_next_token_map = draft_first_token_map

        draft_cur_L = 0
        draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae,
                                          self.draft_model.patch_nums[-1],
                                          self.draft_model.patch_nums[-1])

        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)


        draft_token_hub = []
        
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            
            if si >= entry_num:
                break

            ratio = si / self.num_stages_minus_1
            draft_cur_L += pn*pn
            x = draft_next_token_map
            
            AdaLNSelfAttn.forward
            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)            
            
            t = cfg * ratio
            draft_logits_BlV = (1+t)*draft_logits_BlV[:B] - t*draft_logits_BlV[B:]  # (B, l, V)

            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV,
                rng=self.draft_model.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            if not more_smooth:
                draft_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), tau=draft_gum_t, hard=False, dim=-1, rng=self.draft_model.rng
                    ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            draft_h_BChw = draft_h_BChw.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, draft_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, draft_h_BChw
            )

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1,2)
                draft_token_hub.append(draft_next_token_map)
                draft_next_token_map = (
                    self.draft_model.word_embed(draft_next_token_map)
                    + draft_lvl_pos[:, draft_cur_L : draft_cur_L + next_pn*next_pn]
                )
                draft_next_token_map = draft_next_token_map.repeat(2,1,1)

            if si == self.num_stages_minus_1:
                for blk in self.draft_model.blocks:
                    blk.attn.kv_caching(False)
                return self.draft_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

        # draft模型生成完毕  
        if len(draft_token_hub) != 0:   
            draft_token_hub = torch.cat(draft_token_hub, dim = 1)      
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        
    
        ###### target模型接受draft模型生成的内容然后生成最后一层的内容
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        pindex = exit_points[entry_num]

        device = torch.device("cuda:0")
        attn_bias = attn_bias_for_sdmasking[:,:,0:pindex,0:pindex]
        attn_bias = attn_bias.to(device)

        self.target_model.rng = self.draft_model.rng
        target_label_B = draft_label_B

        target_sos = target_cond_BD = self.target_model.class_emb(
            torch.cat((target_label_B, torch.full_like(target_label_B, fill_value=self.target_model.num_classes)), dim=0)
        )

        assert torch.equal(target_sos, draft_sos)
        target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC

        assert torch.equal(target_lvl_pos, draft_lvl_pos)
        
        target_first_token_map = target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1) \
            + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1) \
            + target_lvl_pos[:, :self.target_model.first_l]

        assert(target_first_token_map, draft_first_token_map)
        target_cur_L = 0
        target_f_hat =  draft_f_hat


        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)

        assert torch.equal(target_cond_BD_or_gss, draft_cond_BD_or_gss)
        if not len(draft_token_hub) == 0:
            # 接受之前生成的做为target_model输出的prefix
            target_next_token_map = draft_token_hub    

            target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:,1:pindex]  
            
            # 正常来说前边的已经进行过调整，所以这里应该只有最后一段需要cfg的修改。
            target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            if len(target_next_token_map) != 0:
                target_next_token_map = torch.cat([target_first_token_map,target_next_token_map],dim=1)
            else:
                target_next_token_map = torch.cat([target_first_token_map],dim=1)
            

            target_f_hat = target_sos.new_zeros(B, self.target_model.Cvae,
                                          self.target_model.patch_nums[-1],
                                          self.target_model.patch_nums[-1])
        else: 
            target_next_token_map = target_first_token_map
        
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            target_cur_L += pn*pn
            t = cfg * ratio 
            
            if si < entry_num:
                continue

            # 我们实际上只需要让进入那一层找到对应的next_token_map就可以了，剩下的就是x = target_next_token_map
            # sd_mask = 1,我们使用自己写的掩码
            if sd_mask == 1:
                x = target_next_token_map
                AdaLNSelfAttn.forward
                # 这里我们暂时不检测也不用attn_bias，因为我们当前只截取了进入层的
                if si == entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
                else:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)

                if si == entry_num:
                    x = target_next_token_map[:, exit_points[entry_num-1]:pindex]
                    target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                else:
                    target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                    
                if si == entry_num:
                    x = target_next_token_map[:, exit_points[entry_num-1]:pindex]
                else:
                    x = target_next_token_map
                AdaLNSelfAttn.forward
                if si >= entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
            # sd_mask = 0, 不需要使用掩码
            else:
                if si == entry_num:
                    x = target_next_token_map[:, exit_points[entry_num-1]:pindex]
                else:
                    x = target_next_token_map
                AdaLNSelfAttn.forward
                if si >= entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)

            # 这里进行了改动，我们没有进行重新采样，因为实际上我们应该继续使用之前的f_hat,

            target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
            target_idx_Bl = sample_with_top_k_top_p_(
                target_logits_BlV,
                rng=self.target_model.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]


            if not more_smooth: # this is the default case
                target_h_BChw = self.target_model.vae_quant_proxy[0].embedding(target_idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio),
                    tau=target_gum_t,
                    hard=False, dim=-1,
                    rng=self.target_model.rng
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose_(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), target_f_hat, target_h_BChw
            )
            
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            
        # target模型生成完成
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)   
                    
        return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    # 是否使用sd_masking

    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_test1(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        entry_num: int = 10, 
        sd_mask: int = 0
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        这里可以考虑top_k, top_p是否需要将target_model和draft_model分开，这样可以更加有效？
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param entry_num: 转换模型的节点
        :param sd_mask: 是否使用我们自己写的block_wise的掩码
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        
        ###### 初始化参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1

        total_stages = len(self.patch_nums)

        if g_seed is not None:
            self.draft_model.rng.manual_seed(g_seed)
        else:
            self.draft_model.rng = None

        draft_label_B = label_B
        if draft_label_B is None:
            draft_label_B = torch.multinomial(
                self.draft_model.uniform_prob, num_samples=B, replacement=True, generator=self.draft_model.rng
            ).reshape(B)
        elif isinstance(draft_label_B, int):
            draft_label_B = torch.full(
                (B,),
                fill_value=self.draft_model.num_classes if draft_label_B < 0 else draft_label_B,
                device=self.draft_model.lvl_1L.device
            )

        draft_sos = draft_cond_BD = self.draft_model.class_emb(
            torch.cat((draft_label_B, torch.full_like(draft_label_B, fill_value=self.draft_model.num_classes)), dim=0)
        )   

        draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC

        draft_first_token_map = (
            draft_sos.unsqueeze(1).expand(2*B, self.draft_model.first_l, -1)
            + self.draft_model.pos_start.expand(2*B, self.draft_model.first_l, -1)
            + draft_lvl_pos[:, :self.draft_model.first_l]
        )

        draft_next_token_map = draft_first_token_map

        draft_cur_L = 0
        draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae,
                                          self.draft_model.patch_nums[-1],
                                          self.draft_model.patch_nums[-1])

        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)


        draft_token_hub = []
        
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            
            # 生成0-entry_num-1
            if si >= entry_num:
                break

            ratio = si / self.num_stages_minus_1
            draft_cur_L += pn*pn
            x = draft_next_token_map
            
            AdaLNSelfAttn.forward
            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)            
            
            t = cfg * ratio
            # print(f"draft:{draft_logits_BlV.shape}")
            draft_logits_BlV = (1+t)*draft_logits_BlV[:B] - t*draft_logits_BlV[B:]  # (B, l, V)

            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV,
                rng=self.draft_model.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            if not more_smooth:
                draft_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), tau=draft_gum_t, hard=False, dim=-1, rng=self.draft_model.rng
                    ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            draft_h_BChw = draft_h_BChw.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, draft_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, draft_h_BChw
            )

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1,2)
                draft_token_hub.append(draft_next_token_map)
                draft_next_token_map = (
                    self.draft_model.word_embed(draft_next_token_map)
                    + draft_lvl_pos[:, draft_cur_L : draft_cur_L + next_pn*next_pn]
                )
                draft_next_token_map = draft_next_token_map.repeat(2,1,1)

            if si == self.num_stages_minus_1:
                for blk in self.draft_model.blocks:
                    blk.attn.kv_caching(False)
                # print("done")
                return self.draft_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

        # draft模型生成完毕  
        if len(draft_token_hub) != 0:   
            draft_token_hub = torch.cat(draft_token_hub, dim = 1)      
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        
    
        ###### target模型接受draft模型生成的内容然后生成最后一层的内容
        start_points = [0,1,5,14,30,55,91,155,255,424]
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        pindex = exit_points[entry_num]
        sindex = start_points[entry_num]
        device = torch.device("cuda:0")

        attn_bias = self.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex]
        attn_bias = attn_bias.to(device)

        self.target_model.rng = self.draft_model.rng
        target_label_B = draft_label_B

        target_sos = target_cond_BD = self.target_model.class_emb(
            torch.cat((target_label_B, torch.full_like(target_label_B, fill_value=self.target_model.num_classes)), dim=0)
        )

        assert torch.equal(target_sos, draft_sos)
        target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC

        assert torch.equal(target_lvl_pos, draft_lvl_pos)
        
        target_first_token_map = target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1) \
            + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1) \
            + target_lvl_pos[:, :self.target_model.first_l]

        assert torch.equal(target_first_token_map,draft_first_token_map)
        target_cur_L = 0
        target_f_hat = draft_f_hat

        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)

        assert torch.equal(target_cond_BD_or_gss, draft_cond_BD_or_gss)
        # 如果draft_token_hub不为0
        if not len(draft_token_hub) == 0:
            # 接受之前生成的做为target_model输出的prefix
            target_next_token_map = draft_token_hub    

            target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:,1:pindex]  
            
            # 正常来说前边的已经进行过调整，所以这里应该只有最后一段需要cfg的修改。
            target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            if len(target_next_token_map) != 0:
                target_next_token_map = torch.cat([target_first_token_map,target_next_token_map],dim=1)
            else:
                target_next_token_map = target_first_token_map
            

        else: 
            target_next_token_map = target_first_token_map
            target_f_hat = target_sos.new_zeros(B, self.target_model.Cvae,
                                          self.target_model.patch_nums[-1],
                                          self.target_model.patch_nums[-1])       
        
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            target_cur_L += pn*pn
            t = cfg * ratio 
            
            if si < entry_num:
                continue

            # 我们实际上只需要让进入那一层找到对应的next_token_map就可以了，剩下的就是x = target_next_token_map
            # sd_mask = 1,我们使用自己写的掩码
            if sd_mask == 1:
                x = target_next_token_map
                AdaLNSelfAttn.forward
                # 这里我们暂时不检测也不用attn_bias，因为我们当前只截取了进入层的
                if si == entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
                else:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)

                if si == entry_num:
                    x = target_next_token_map[:,sindex:pindex]
                    target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                else:
                    target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
            # sd_mask = 0, 不需要使用掩码
            else:
                if si == entry_num:
                    x = target_next_token_map[:, exit_points[entry_num-1]:pindex]
                else:
                    x = target_next_token_map
                AdaLNSelfAttn.forward
                if si >= entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)

            # 这里进行了改动，我们没有进行重新采样，因为实际上我们应该继续使用之前的f_hat,

            # print(f"target:{target_logits_BlV.shape}")

            target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
            target_idx_Bl = sample_with_top_k_top_p_(
                target_logits_BlV,
                rng=self.target_model.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]


            if not more_smooth: # this is the default case
                target_h_BChw = self.target_model.vae_quant_proxy[0].embedding(target_idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio),
                    tau=target_gum_t,
                    hard=False, dim=-1,
                    rng=self.target_model.rng
                ) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose_(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), target_f_hat, target_h_BChw
            )
            
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            
        # target模型生成完成
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)   
                    
        return self.target_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    
    # 换成同一个vae模型进行解码
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_test2(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        entry_num: int = 10, 
        sd_mask: int = 0
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        这里可以考虑top_k, top_p是否需要将target_model和draft_model分开，这样可以更加有效？
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param entry_num: 转换模型的节点
        :param sd_mask: 是否使用我们自己写的block_wise的掩码
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        
        ###### 初始化参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1

        total_stages = len(self.patch_nums)

        if g_seed is not None:
            self.draft_model.rng.manual_seed(g_seed)
        else:
            self.draft_model.rng = None

        draft_label_B = label_B
        if draft_label_B is None:
            draft_label_B = torch.multinomial(
                self.draft_model.uniform_prob, num_samples=B, replacement=True, generator=self.draft_model.rng
            ).reshape(B)
        elif isinstance(draft_label_B, int):
            draft_label_B = torch.full(
                (B,),
                fill_value=self.draft_model.num_classes if draft_label_B < 0 else draft_label_B,
                device=self.draft_model.lvl_1L.device
            )

        draft_sos = draft_cond_BD = self.draft_model.class_emb(
            torch.cat((draft_label_B, torch.full_like(draft_label_B, fill_value=self.draft_model.num_classes)), dim=0)
        )   

        draft_lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC

        draft_first_token_map = (
            draft_sos.unsqueeze(1).expand(2*B, self.draft_model.first_l, -1)
            + self.draft_model.pos_start.expand(2*B, self.draft_model.first_l, -1)
            + draft_lvl_pos[:, :self.draft_model.first_l]
        )

        draft_next_token_map = draft_first_token_map

        draft_cur_L = 0
        draft_f_hat = draft_sos.new_zeros(B, self.draft_model.Cvae,
                                          self.draft_model.patch_nums[-1],
                                          self.draft_model.patch_nums[-1])

        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(draft_cond_BD)


        draft_token_hub = []
        
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            
            # 生成0-entry_num-1
            if si >= entry_num:
                break

            ratio = si / self.num_stages_minus_1
            draft_cur_L += pn*pn
            x = draft_next_token_map
            
            AdaLNSelfAttn.forward
            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)            
            
            t = cfg * ratio
            # print(f"draft:{draft_logits_BlV.shape}")
            draft_logits_BlV = (1+t)*draft_logits_BlV[:B] - t*draft_logits_BlV[B:]  # (B, l, V)

            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV,
                rng=self.draft_model.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            if not more_smooth:
                draft_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), tau=draft_gum_t, hard=False, dim=-1, rng=self.draft_model.rng
                    ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            draft_h_BChw = draft_h_BChw.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, draft_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, draft_h_BChw
            )

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                draft_next_token_map = draft_next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1,2)
                draft_token_hub.append(draft_next_token_map)
                draft_next_token_map = (
                    self.draft_model.word_embed(draft_next_token_map)
                    + draft_lvl_pos[:, draft_cur_L : draft_cur_L + next_pn*next_pn]
                )
                draft_next_token_map = draft_next_token_map.repeat(2,1,1)

            if si == self.num_stages_minus_1:
                for blk in self.draft_model.blocks:
                    blk.attn.kv_caching(False)
                # print("done")
                return self.draft_model.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

        # draft模型生成完毕  
        if len(draft_token_hub) != 0:   
            draft_token_hub = torch.cat(draft_token_hub, dim = 1)      
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        
    
        ###### target模型接受draft模型生成的内容然后生成最后一层的内容
        start_points = [0,1,5,14,30,55,91,155,255,424]
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        pindex = exit_points[entry_num]
        sindex = start_points[entry_num]
        device = torch.device("cuda:0")

        attn_bias = self.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex]
        attn_bias = attn_bias.to(device)

        self.target_model.rng = self.draft_model.rng
        target_label_B = draft_label_B

        target_sos = target_cond_BD = self.target_model.class_emb(
            torch.cat((target_label_B, torch.full_like(target_label_B, fill_value=self.target_model.num_classes)), dim=0)
        )

        # assert torch.equal(target_sos, draft_sos)
        target_lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC

        # assert torch.equal(target_lvl_pos, draft_lvl_pos)
        
        target_first_token_map = target_sos.unsqueeze(1).expand(2 * B, self.target_model.first_l, -1) \
            + self.target_model.pos_start.expand(2 * B, self.target_model.first_l, -1) \
            + target_lvl_pos[:, :self.target_model.first_l]

        # assert torch.equal(target_first_token_map,draft_first_token_map)
        target_cur_L = 0
        target_f_hat = draft_f_hat

        target_cond_BD_or_gss = self.target_model.shared_ada_lin(target_cond_BD)

        # assert torch.equal(target_cond_BD_or_gss, draft_cond_BD_or_gss)
        # 如果draft_token_hub不为0
        if not len(draft_token_hub) == 0:
            # 接受之前生成的做为target_model输出的prefix
            target_next_token_map = draft_token_hub    

            target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:,1:pindex]  
            
            # 正常来说前边的已经进行过调整，所以这里应该只有最后一段需要cfg的修改。
            target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            if len(target_next_token_map) != 0:
                target_next_token_map = torch.cat([target_first_token_map,target_next_token_map],dim=1)
            else:
                target_next_token_map = target_first_token_map
            

        else: 
            target_next_token_map = target_first_token_map
            target_f_hat = target_sos.new_zeros(B, self.target_model.Cvae,
                                          self.target_model.patch_nums[-1],
                                          self.target_model.patch_nums[-1])       
        
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            target_cur_L += pn*pn
            t = cfg * ratio 
            
            if si < entry_num:
                continue

            # 我们实际上只需要让进入那一层找到对应的next_token_map就可以了，剩下的就是x = target_next_token_map
            # sd_mask = 1,我们使用自己写的掩码
            if sd_mask == 1:
                x = target_next_token_map
                AdaLNSelfAttn.forward
                # 这里我们暂时不检测也不用attn_bias，因为我们当前只截取了进入层的
                if si == entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
                else:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)

                if si == entry_num:
                    x = target_next_token_map[:,sindex:pindex]
                    target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                else:
                    target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
            # sd_mask = 0, 不需要使用掩码
            else:
                if si == entry_num:
                    x = target_next_token_map[:, sindex:pindex]
                else:
                    x = target_next_token_map
                AdaLNSelfAttn.forward
                if si >= entry_num:
                    for b in self.target_model.blocks:
                        x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)

            # 这里进行了改动，我们没有进行重新采样，因为实际上我们应该继续使用之前的f_hat,

            # print(f"target:{target_logits_BlV.shape}")

            target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
            target_idx_Bl = sample_with_top_k_top_p_(
                target_logits_BlV,
                rng=self.target_model.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]


            if not more_smooth: # this is the default case
                target_h_BChw = self.draft_model.vae_quant_proxy[0].embedding(target_idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio),
                    tau=target_gum_t,
                    hard=False, dim=-1,
                    rng=self.draft_model.rng
                ) @ self.draft_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose_(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), target_f_hat, target_h_BChw
            )
            
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                target_next_token_map = self.target_model.word_embed(target_next_token_map) + target_lvl_pos[:, target_cur_L:target_cur_L + next_pn * next_pn]
                target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            
        # target模型生成完成
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)   
                    
        return self.draft_model.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
