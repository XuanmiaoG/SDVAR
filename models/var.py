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
        
        self.attn_bias_for_sdmasking = attn_bias_for_sdmasking

    def init_param(
            self,
            model: VAR,
            B: int,
            label_B,
        ):        
        sos = cond_BD = model.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=model.num_classes)), dim=0)
        )   
        cond_BD_or_gss = model.shared_ada_lin(cond_BD)

        lvl_pos = model.lvl_embed(model.lvl_1L) + model.pos_1LC

        first_token_map = (
            sos.unsqueeze(1).expand(2*B, model.first_l, -1)
            + model.pos_start.expand(2*B, model.first_l, -1)
            + lvl_pos[:, :model.first_l]
        )

        first_f_hat = sos.new_zeros(B, model.Cvae, model.patch_nums[-1], model.patch_nums[-1])

        return sos, cond_BD, cond_BD_or_gss, lvl_pos, first_token_map, first_f_hat

    # 初始化分离和选择掩码方式
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_test3(
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
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param entry_num: 转换模型的节点
        :param sd_mask: 是否使用我们自己写的block_wise的掩码
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        ###### 通用参数参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1

        total_stages = len(self.patch_nums)

        self.vae_proxy = self.target_model.vae_proxy
        self.vae_quant_proxy = self.target_model.vae_quant_proxy

        if g_seed is not None:
            self.rng = self.target_model.rng.manual_seed(g_seed)
        else:
            self.rng = None
        

        if label_B is None:
            label_B = torch.multinomial(
                self.target_model.uniform_prob, num_samples=B, replacement=True, generator=self.rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.target_model.num_classes if label_B < 0 else label_B,
                device=self.target_model.lvl_1L.device
            )

        draft_sos, draft_cond_BD, draft_cond_BD_or_gss, \
        draft_lvl_pos, draft_first_token_map, draft_f_hat = self.init_param(self.draft_model, B, label_B)

        draft_cur_L = 0
        draft_next_token_map = draft_first_token_map
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
            draft_logits_BlV = (1+t)*draft_logits_BlV[:B] - t*draft_logits_BlV[B:]  # (B, l, V)

            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV,
                rng=self.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            # 在所有模型中都使用同一个vae
            if not more_smooth:
                draft_h_BChw = self.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), tau=draft_gum_t, hard=False, dim=-1, rng=self.rng
                    ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            draft_h_BChw = draft_h_BChw.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, draft_next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
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
                return self.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

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


        target_sos, target_cond_BD, target_cond_BD_or_gss, \
        target_lvl_pos, target_first_token_map, target_f_hat = self.init_param(self.target_model, B, label_B)

        target_cur_L = 0
        target_f_hat = draft_f_hat

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
        
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            target_cur_L += pn*pn
            t = cfg * ratio 
            
            if si < entry_num:
                continue

            # 我们实际上只需要让进入那一层找到对应的next_token_map就可以了，剩下的就是x = target_next_token_map
            if sd_mask != 0:
                if sd_mask == 1:
                    # sd_mask = 1, 全部层包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex]
                    attn_bias = attn_bias.to(device)
                if sd_mask == 2:
                    # sd_mask = 2, 全部层不包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                if sd_mask == 3:
                    # sd_mask = 3, 进行因果掩码
                    attn_bias = self.target_model.attn_bias_for_masking[:,:,0:pindex,0:pindex]

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
                    
            else:
                # sd_mask = 0, 不需要使用掩码
                if si == entry_num:
                    x = target_next_token_map[:,sindex:pindex]
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
                rng=self.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]


            if not more_smooth: # this is the default case
                target_h_BChw = self.vae_quant_proxy[0].embedding(target_idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio),
                    tau=target_gum_t,
                    hard=False, dim=-1,
                    rng=self.rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose_(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
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
                    
        return self.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
        
    # 增加warmup环节，默认只用1轮的warmup
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_test4(
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
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param entry_num: 转换模型的节点
        :param sd_mask: 是否使用我们自己写的block_wise的掩码
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        ###### 通用参数参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1

        total_stages = len(self.patch_nums)

        self.vae_proxy = self.target_model.vae_proxy
        self.vae_quant_proxy = self.target_model.vae_quant_proxy

        if g_seed is not None:
            self.rng = self.target_model.rng.manual_seed(g_seed)
        else:
            self.rng = None
        

        if label_B is None:
            label_B = torch.multinomial(
                self.target_model.uniform_prob, num_samples=B, replacement=True, generator=self.rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.target_model.num_classes if label_B < 0 else label_B,
                device=self.target_model.lvl_1L.device
            )
        start_points = [0,1,5,14,30,55,91,155,255,424]
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        device = torch.device("cuda:0")

        #####
        # target_model生成warmup
        #####
        warmup_step = 0

        target_sos, target_cond_BD, target_cond_BD_or_gss, \
        target_lvl_pos, target_first_token_map, target_f_hat = self.init_param(self.target_model, B, label_B)

        target_cur_L = 0
        target_next_token_map = target_first_token_map

        warmup_token_hub =  []

        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            
            # 只生成到warmup_step
            if si >= warmup_step + 1:
                break;

            ratio = si / self.num_stages_minus_1
            target_cur_L += pn*pn
            x = target_next_token_map
            
            AdaLNSelfAttn.forward
            for blk in self.target_model.blocks:
                x = blk(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
            target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)            
            
            t = cfg * ratio
            target_logits_BlV = (1+t)*target_logits_BlV[:B] - t*target_logits_BlV[B:]  # (B, l, V)

            target_idx_Bl = sample_with_top_k_top_p_(
                target_logits_BlV,
                rng=self.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            # 在所有模型中都使用同一个vae
            if not more_smooth:
                target_h_BChw = self.vae_quant_proxy[0].embedding(target_idx_Bl)
            else:
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio), tau=target_gum_t, hard=False, dim=-1, rng=self.rng
                    ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            target_h_BChw = target_h_BChw.transpose(1,2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, target_f_hat, target_h_BChw
            )

            if si != self.num_stages_minus_1:   # prepare for next stage
                next_pn = self.patch_nums[si+1]
                target_next_token_map = target_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1,2)
                warmup_token_hub.append(target_next_token_map)
                target_next_token_map = (
                    self.target_model.word_embed(target_next_token_map)
                    + target_lvl_pos[:, target_cur_L : target_cur_L + next_pn*next_pn]
                )
                target_next_token_map = target_next_token_map.repeat(2,1,1)

            if si == self.num_stages_minus_1:
                for blk in self.target_model.blocks:
                    blk.attn.kv_caching(False)
                return self.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

        warmup_token_hub = torch.cat(warmup_token_hub, dim = 1)      
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)

        #####
        # draft_model生成
        ##### 
        draft_sos, draft_cond_BD, draft_cond_BD_or_gss, \
        draft_lvl_pos, draft_first_token_map, draft_f_hat = self.init_param(self.draft_model, B, label_B)

        draft_cur_L = 0
        draft_next_token_map = draft_first_token_map
        draft_token_hub = []
        pindex = exit_points[warmup_step+1] 
        sindex = start_points[warmup_step]
        
        # 继承warmup
        draft_next_token_map = warmup_token_hub
        draft_next_token_map = self.draft_model.word_embed(draft_next_token_map) + draft_lvl_pos[:,1:5]  
        draft_next_token_map = draft_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        draft_next_token_map = torch.cat([draft_first_token_map,draft_next_token_map],dim=1)
 
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):
            # 生成warmup的要跳过
            if si <= warmup_step:
                continue

            # 生成1-entry_num-1
            if si >= entry_num:
                break

            ratio = si / self.num_stages_minus_1
            draft_cur_L += pn*pn

            if sd_mask != 0:

                if sd_mask == 1:
                    # sd_mask = 1, 全部层包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex]
                    attn_bias = attn_bias.to(device)
                if sd_mask == 2:
                    # sd_mask = 2, 全部层不包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                if sd_mask == 3:
                    # sd_mask = 3, 进行因果掩码
                    attn_bias = self.draft_model.attn_bias_for_masking[:,:,0:pindex,0:pindex]

                x = draft_next_token_map
                AdaLNSelfAttn.forward
                # 这里我们暂时不检测也不用attn_bias，因为我们当前只截取了进入层的
                if si == entry_num:
                    for b in self.draft_model.blocks:
                        x = b(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=attn_bias)
                else:
                    for b in self.draft_model.blocks:
                        x = b(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)

                if si == entry_num:
                    x = draft_next_token_map[:,sindex:pindex]
                    draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)
                else:
                    draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)

            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)            
            
            t = cfg * ratio
            draft_logits_BlV = (1+t)*draft_logits_BlV[:B] - t*draft_logits_BlV[B:]  # (B, l, V)

            draft_idx_Bl = sample_with_top_k_top_p_(
                draft_logits_BlV,
                rng=self.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]

            # 在所有模型中都使用同一个vae
            if not more_smooth:
                draft_h_BChw = self.vae_quant_proxy[0].embedding(draft_idx_Bl)
            else:
                draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                draft_h_BChw = gumbel_softmax_with_rng(
                    draft_logits_BlV.mul(1 + ratio), tau=draft_gum_t, hard=False, dim=-1, rng=self.rng
                    ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            draft_h_BChw = draft_h_BChw.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, draft_next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
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
                return self.vae_proxy[0].fhat_to_img(draft_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

        # draft模型生成完毕  
        if len(draft_token_hub) != 0:   
            draft_token_hub = torch.cat(draft_token_hub, dim = 1)      
            draft_token_hub = torch.cat((warmup_token_hub,draft_token_hub), dim = 1)
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
        
    
        ######
        #  target模型接受draft模型生成的内容然后生成最后一层的内容
        ######

        pindex = exit_points[entry_num]
        sindex = start_points[entry_num]

        # 由于我们warmup已经做了初始化所以就不用再整一遍了
        # target_sos, target_cond_BD, target_cond_BD_or_gss, \
        # target_lvl_pos, target_first_token_map, target_f_hat = self.init_param(self.target_model, B, label_B)

        target_cur_L = 0
        target_f_hat = draft_f_hat

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
        
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            target_cur_L += pn*pn
            t = cfg * ratio 
            
            if si < entry_num:
                continue

            # 我们实际上只需要让进入那一层找到对应的next_token_map就可以了，剩下的就是x = target_next_token_map
            if sd_mask != 0:
                if sd_mask == 1:
                    # sd_mask = 1, 全部层包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_sdmasking[:,:,0:pindex,0:pindex]
                    attn_bias = attn_bias.to(device)
                if sd_mask == 2:
                    # sd_mask = 2, 全部层不包括未预测这层进行block-wise的掩码
                    attn_bias = self.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                if sd_mask == 3:
                    # sd_mask = 3, 进行因果掩码
                    attn_bias = self.target_model.attn_bias_for_masking[:,:,0:pindex,0:pindex]

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
                    
            else:
                # sd_mask = 0, 不需要使用掩码
                if si == entry_num:
                    x = target_next_token_map[:,sindex:pindex]
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
                rng=self.rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1
            )[:, :, 0]


            if not more_smooth: # this is the default case
                target_h_BChw = self.vae_quant_proxy[0].embedding(target_idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                target_h_BChw = gumbel_softmax_with_rng(
                    target_logits_BlV.mul(1 + ratio),
                    tau=target_gum_t,
                    hard=False, dim=-1,
                    rng=self.rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            target_h_BChw = target_h_BChw.transpose_(1, 2).reshape(B, self.target_model.Cvae, pn, pn)

            target_f_hat, target_next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
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
                    
        return self.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_sd_speculative(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        entry_num: int = 10,
        sd_mask: int = 0,
        similarity_threshold: float = 0.8,
        draft_steps: int = 2,  # 每次draft模型预测的步数
        verification_k: int = 5  # top-k验证参数
    ):
        """
        Speculative decoding implementation for VAR model with modular design
        """
        # 对于小的entry_num直接使用目标模型
        if entry_num <= 1:
            return self.target_model.autoregressive_infer_cfg(
                B=B, label_B=label_B, g_seed=g_seed, cfg=cfg,
                top_k=top_k, top_p=top_p, more_smooth=more_smooth
            )
        
        # 初始化共同参数
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1
        total_stages = len(self.patch_nums)
        
        start_points = [0, 1, 5, 14, 30, 55, 91, 155, 255, 424]
        exit_points = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
        device = torch.device("cuda:0")
        
        self.vae_proxy = self.target_model.vae_proxy
        self.vae_quant_proxy = self.target_model.vae_quant_proxy
        
        if g_seed is not None:
            self.rng = self.target_model.rng.manual_seed(g_seed)
        else:
            self.rng = None
        
        # 处理标签批次
        if label_B is None:
            label_B = torch.multinomial(
                self.target_model.uniform_prob, num_samples=B, replacement=True, generator=self.rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.target_model.num_classes if label_B < 0 else label_B,
                device=self.target_model.lvl_1L.device
            )
        
        # 第1阶段：使用目标模型进行暖身步骤（warmup_steps）
        # 初始化模型参数
        warmup_steps = min(entry_num - 1, 2)  # 使用至少1个但不超过entry_num-1的暖身步骤
        
        print(f"执行{warmup_steps}步暖身...")
        
        # 初始化target模型参数
        target_sos, target_cond_BD, target_cond_BD_or_gss, \
        target_lvl_pos, target_first_token_map, target_f_hat = self.init_param(self.target_model, B, label_B)
        
        # 辅助函数：目标模型生成n步
        def target_generate_steps(start_step, num_steps, next_token_map, f_hat):
            """目标模型连续生成num_steps步的token"""
            current_token_map = next_token_map
            current_f_hat = f_hat
            token_hub = []
            f_hat_history = []
            token_id_history = []
            
            # 打开KV缓存
            for blk in self.target_model.blocks:
                blk.attn.kv_caching(True)
            
            for i in range(num_steps):
                si = start_step + i
                if si >= total_stages:
                    break
                    
                pn = self.patch_nums[si]
                ratio = si / self.num_stages_minus_1
                t = cfg * ratio
                
                # 处理当前token
                x = current_token_map
                
                # 通过transformer块
                for b in self.target_model.blocks:
                    x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=None)
                
                # 获取logits并应用CFG
                target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
                target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
                
                # 采样token
                target_idx_Bl = sample_with_top_k_top_p_(
                    target_logits_BlV,
                    rng=self.rng,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1
                )[:, :, 0]
                
                # 保存token ID
                token_id_history.append(target_idx_Bl.clone())
                
                # 转换为嵌入向量
                if not more_smooth:
                    target_h_BChw = self.vae_quant_proxy[0].embedding(target_idx_Bl)
                else:
                    target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    target_h_BChw = gumbel_softmax_with_rng(
                        target_logits_BlV.mul(1 + ratio),
                        tau=target_gum_t,
                        hard=False,
                        dim=-1,
                        rng=self.rng
                    ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                
                # 转换形状并处理
                target_h_BChw = target_h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, pn, pn)
                current_f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, total_stages, current_f_hat, target_h_BChw
                )
                
                # 保存状态历史
                f_hat_history.append(current_f_hat.clone())
                
                # 准备下一阶段
                if si != self.num_stages_minus_1:
                    next_pn = self.patch_nums[si+1]
                    next_token_map = next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                    token_hub.append(next_token_map.clone())
                    
                    current_token_map = self.target_model.word_embed(next_token_map) + \
                        target_lvl_pos[:, (si+1==0)*0 + (si+1>0)*(start_points[si+1 if si+1 < len(start_points) else -1])]
                    current_token_map = current_token_map.repeat(2, 1, 1)
                
            # 关闭KV缓存
            for blk in self.target_model.blocks:
                blk.attn.kv_caching(False)
            
            if len(token_hub) > 0:
                token_hub = torch.cat(token_hub, dim=1)
            
            return token_hub, current_f_hat, f_hat_history, token_id_history, next_token_map
        
        # 辅助函数：draft模型生成n步
        def draft_generate_steps(start_step, num_steps, next_token_map, f_hat):
            """draft模型连续生成num_steps步的token"""
            # 初始化draft模型参数
            draft_sos, draft_cond_BD, draft_cond_BD_or_gss, \
            draft_lvl_pos, _, _ = self.init_param(self.draft_model, B, label_B)
            
            current_token_map = next_token_map
            current_f_hat = f_hat
            token_hub = []
            input_token_history = []
            f_hat_history = []
            token_id_history = []
            
            # 打开KV缓存
            for blk in self.draft_model.blocks:
                blk.attn.kv_caching(True)
            
            for i in range(num_steps):
                si = start_step + i
                if si >= total_stages:
                    break
                    
                pn = self.patch_nums[si]
                ratio = si / self.num_stages_minus_1
                t = cfg * ratio
                
                # 保存输入token用于验证
                input_token_history.append(current_token_map.clone())
                
                # 处理当前token
                x = current_token_map
                
                # 通过transformer块
                for blk in self.draft_model.blocks:
                    x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
                
                # 获取logits并应用CFG
                draft_logits_BlV = self.draft_model.get_logits(x, draft_cond_BD)
                draft_logits_BlV = (1+t) * draft_logits_BlV[:B] - t * draft_logits_BlV[B:]
                
                # 采样token
                draft_idx_Bl = sample_with_top_k_top_p_(
                    draft_logits_BlV,
                    rng=self.rng,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1
                )[:, :, 0]
                
                # 保存token ID
                token_id_history.append(draft_idx_Bl.clone())
                
                # 转换为嵌入向量
                if not more_smooth:
                    draft_h_BChw = self.vae_quant_proxy[0].embedding(draft_idx_Bl)
                else:
                    draft_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    draft_h_BChw = gumbel_softmax_with_rng(
                        draft_logits_BlV.mul(1 + ratio),
                        tau=draft_gum_t,
                        hard=False,
                        dim=-1,
                        rng=self.rng
                    ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                
                # 在转换形状前打印维度信息用于调试
                print(f"draft_h_BChw shape: {draft_h_BChw.shape}, pn: {pn}, target shape: [B={B}, C={self.draft_model.Cvae}, pn={pn}, pn={pn}]")

                # 确保形状匹配
                expected_size = B * self.draft_model.Cvae * pn * pn
                actual_size = draft_h_BChw.numel()
                if actual_size != expected_size:
                    print(f"维度不匹配! 期望大小: {expected_size}, 实际大小: {actual_size}")
                    # 临时解决方案：调整pn以匹配实际大小
                    adjusted_pn = int(math.sqrt(actual_size / (B * self.draft_model.Cvae)))
                    print(f"调整pn为: {adjusted_pn}")
                    pn = adjusted_pn

                # 现在使用可能调整过的pn进行转换
                draft_h_BChw = draft_h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)
                current_f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, total_stages, current_f_hat, draft_h_BChw
                )
                
                # 保存状态历史
                f_hat_history.append(current_f_hat.clone())
                
                # 准备下一阶段
                if si != self.num_stages_minus_1 and i < num_steps - 1:
                    next_pn = self.patch_nums[si+1]
                    next_token_map = next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1, 2)
                    token_hub.append(next_token_map.clone())
                    
                    current_token_map = self.draft_model.word_embed(next_token_map) + \
                        draft_lvl_pos[:, (si+1==0)*0 + (si+1>0)*(start_points[si+1 if si+1 < len(start_points) else -1])]
                    current_token_map = current_token_map.repeat(2, 1, 1)
            
            # 关闭KV缓存
            for blk in self.draft_model.blocks:
                blk.attn.kv_caching(False)
            
            if len(token_hub) > 0:
                token_hub = torch.cat(token_hub, dim=1)
            
            return input_token_history, f_hat_history, token_hub, token_id_history, next_token_map, current_f_hat
        
        # 辅助函数：目标模型验证draft模型的token
        def target_verify_tokens(current_step, draft_tokens, current_token_map, f_hat):
            """目标模型验证draft模型生成的tokens"""
            # 打开KV缓存
            for blk in self.target_model.blocks:
                blk.attn.kv_caching(True)
            
            # 处理当前token
            x = current_token_map
            
            # 配置mask
            if sd_mask != 0:
                pindex = exit_points[current_step if current_step < len(exit_points) else -1]
                sindex = start_points[current_step if current_step < len(start_points) else -1]
                
                if sd_mask == 1:
                    attn_bias = self.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].to(device)
                elif sd_mask == 2:
                    attn_bias = self.attn_bias_for_sdmasking[:, :, 0:pindex, 0:pindex].clone()
                    attn_bias[:, :, sindex:pindex, :] = 0.0
                    attn_bias = attn_bias.to(device)
                elif sd_mask == 3:
                    attn_bias = self.target_model.attn_bias_for_masking[:, :, 0:pindex, 0:pindex]
            else:
                attn_bias = None
            
            # 通过transformer块
            for b in self.target_model.blocks:
                x = b(x=x, cond_BD=target_cond_BD_or_gss, attn_bias=attn_bias)
            
            # 获取当前步骤的logits
            target_logits_BlV = self.target_model.get_logits(x, target_cond_BD)
            
            # 应用CFG
            ratio = current_step / self.num_stages_minus_1
            t = cfg * ratio
            target_logits_BlV = (1+t) * target_logits_BlV[:B] - t * target_logits_BlV[B:]
            
            # 关闭KV缓存
            for blk in self.target_model.blocks:
                blk.attn.kv_caching(False)
            
            # 计算验证结果
            target_probs = torch.softmax(target_logits_BlV, dim=-1)
            
            # Top-k验证
            topk_values, topk_indices = torch.topk(target_probs, k=verification_k, dim=-1)
            is_in_topk = (topk_indices == draft_tokens.unsqueeze(-1)).any(dim=-1)
            
            # 计算目标模型给draft token的概率
            draft_probs = torch.gather(target_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            prob_match = draft_probs > similarity_threshold
            
            # 合并验证结果：top-k或概率阈值任一满足即可接受
            verification_mask = is_in_topk | prob_match
            
            # 计算接受率
            acceptance_rate = verification_mask.float().mean().item()
            
            # 创建验证后的token：对于验证失败的位置，使用目标模型的预测
            verified_tokens = torch.where(
                verification_mask,
                draft_tokens,                           # 保留通过验证的draft tokens
                torch.argmax(target_probs, dim=-1)      # 对未通过验证的位置使用target模型的预测
            )
            
            return verified_tokens, target_logits_BlV, acceptance_rate, verification_mask
        
        # 开始生成过程
        current_step = 0
        
        # 第1阶段：使用target模型暖身
        warmup_token_hub, target_f_hat, warmup_f_hat_history, warmup_token_id_history, next_token_map = \
            target_generate_steps(current_step, warmup_steps, target_first_token_map, target_f_hat)
        
        current_step += warmup_steps
        
        # 准备下一阶段的token map
        if len(warmup_token_hub) > 0:
            size_needed = start_points[current_step] if current_step < len(start_points) else 0
            
            if warmup_token_hub.size(1) < size_needed:
                # 填充不足的部分
                padding = torch.zeros(B, size_needed - warmup_token_hub.size(1), warmup_token_hub.size(2), 
                                    device=warmup_token_hub.device)
                warmup_token_hub = torch.cat([warmup_token_hub, padding], dim=1)
            elif warmup_token_hub.size(1) > size_needed:
                # 截取所需部分
                warmup_token_hub = warmup_token_hub[:, :size_needed]
            
            target_next_token_map = self.target_model.word_embed(warmup_token_hub) + \
                target_lvl_pos[:, 1:size_needed+1]
            target_next_token_map = target_next_token_map.repeat(2, 1, 1)
            target_next_token_map = torch.cat([target_first_token_map, target_next_token_map], dim=1)
        else:
            target_next_token_map = target_first_token_map
        
        print(f"暖身完成，当前步骤: {current_step}/{total_stages}")
        
        # 第2阶段：推测解码主循环
        while current_step < total_stages:
            # 计算当前批次可预测的步数
            steps_to_predict = min(draft_steps, total_stages - current_step)
            if steps_to_predict <= 0:
                break
                
            print(f"使用draft模型预测 {steps_to_predict} 步...")
            
            # 使用draft模型生成几个步骤
            draft_input_token_history, draft_f_hat_history, draft_token_hub, draft_token_id_history, \
            draft_next_token_map, draft_f_hat = draft_generate_steps(
                current_step, steps_to_predict, target_next_token_map, target_f_hat
            )
            
            # 使用target模型验证
            accepted_steps = 0
            for i in range(steps_to_predict):
                si = current_step + i
                pn = self.patch_nums[si]
                
                print(f"验证步骤 {si}...")
                
                if i == 0:
                    # 第一步使用当前token map
                    verification_token_map = target_next_token_map
                else:
                    # 后续步骤使用上一步生成的next_token_map
                    prev_token = draft_token_id_history[i-1]
                    ratio = (si-1) / self.num_stages_minus_1
                    
                    if not more_smooth:
                        h_BChw = self.vae_quant_proxy[0].embedding(prev_token)
                    else:
                        gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                        h_BChw = gumbel_softmax_with_rng(
                            torch.zeros_like(prev_token).unsqueeze(-1).expand(-1, -1, self.target_model.V) \
                                .scatter_(-1, prev_token.unsqueeze(-1), 1.0).mul(1 + ratio),
                            tau=gum_t, hard=False, dim=-1, rng=self.rng
                        ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                    
                    prev_pn = self.patch_nums[si-1]
                    h_BChw = h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, prev_pn, prev_pn)
                    _, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                        si-1, total_stages, draft_f_hat_history[i-1], h_BChw
                    )
                    
                    next_pn = self.patch_nums[si]
                    next_token_map = next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                    verification_token_map = self.target_model.word_embed(next_token_map) + \
                        target_lvl_pos[:, start_points[si if si < len(start_points) else -1]]
                    verification_token_map = verification_token_map.repeat(2, 1, 1)
                
                # 验证当前步骤
                verified_tokens, _, acceptance_rate, verification_mask = target_verify_tokens(
                    si, draft_token_id_history[i], verification_token_map, 
                    draft_f_hat_history[i-1] if i > 0 else target_f_hat
                )
                
                print(f"  步骤 {si} 接受率: {acceptance_rate*100:.2f}%")
                
                if acceptance_rate >= similarity_threshold:
                    # 接受当前步骤，继续验证下一步
                    accepted_steps += 1
                    
                    # 最后一步不需要准备next_token_map
                    if i == steps_to_predict - 1:
                        target_f_hat = draft_f_hat_history[i].clone()
                        
                        if si < total_stages - 1:
                            # 准备下一阶段的token map
                            next_pn = self.patch_nums[si+1]
                            token_map = draft_next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                            target_next_token_map = self.target_model.word_embed(token_map) + \
                                target_lvl_pos[:, start_points[si+1 if si+1 < len(start_points) else -1]]
                            target_next_token_map = target_next_token_map.repeat(2, 1, 1)
                else:
                    # 拒绝当前步骤，使用verified_tokens生成正确的next_token_map
                    print(f"  拒绝步骤 {si}，使用验证后的token")
                    
                    # 转换验证后的token为嵌入向量
                    ratio = si / self.num_stages_minus_1
                    
                    if not more_smooth:
                        target_h_BChw = self.vae_quant_proxy[0].embedding(verified_tokens)
                    else:
                        target_gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                        target_h_BChw = gumbel_softmax_with_rng(
                            torch.zeros_like(verified_tokens).unsqueeze(-1).expand(-1, -1, self.target_model.V) \
                                .scatter_(-1, verified_tokens.unsqueeze(-1), 1.0).mul(1 + ratio),
                            tau=target_gum_t, hard=False, dim=-1, rng=self.rng
                        ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                    
                    # 转换形状并处理
                    target_h_BChw = target_h_BChw.transpose(1, 2).reshape(B, self.target_model.Cvae, pn, pn)
                    
                    # 使用正确的上一步状态
                    prev_f_hat = draft_f_hat_history[i-1] if i > 0 else target_f_hat
                    
                    target_f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                        si, total_stages, prev_f_hat, target_h_BChw
                    )
                    
                    # 准备下一阶段的token map
                    if si < total_stages - 1:
                        next_pn = self.patch_nums[si+1]
                        next_token_map = next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                        target_next_token_map = self.target_model.word_embed(next_token_map) + \
                            target_lvl_pos[:, start_points[si+1 if si+1 < len(start_points) else -1]]
                        target_next_token_map = target_next_token_map.repeat(2, 1, 1)
                    
                    # 中断验证循环，从拒绝的步骤之后重新开始
                    break
            
            # 更新步数
            current_step += accepted_steps
            print(f"接受了 {accepted_steps}/{steps_to_predict} 步，当前步骤: {current_step}/{total_stages}")
            
            # 如果所有步骤都被拒绝了，使用target模型生成一步
            if accepted_steps == 0:
                print("所有步骤都被拒绝，使用target模型生成一步")
                _, target_f_hat, _, _, next_token_map = target_generate_steps(
                    current_step, 1, target_next_token_map, target_f_hat
                )
                
                current_step += 1
                
                # 准备下一阶段的token map
                if current_step < total_stages:
                    target_next_token_map = next_token_map
        
        # 生成完成，返回结果
        return self.vae_proxy[0].fhat_to_img(target_f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]