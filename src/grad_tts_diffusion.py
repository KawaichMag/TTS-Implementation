                                                                                       

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_mask(lengths, max_len=None):
                                                                      
    if max_len is None:
        max_len = int(lengths.max().item())
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def length_regulate(hidden_states, durations):
                                                                                  
    expanded = []
    lengths = []
    batch_size, _, channels = hidden_states.shape

    for i in range(batch_size):
        pieces = []
        for state, duration in zip(hidden_states[i], durations[i]):
                                                                            
            repeats = int(max(duration.item(), 0))
            if repeats > 0:
                pieces.append(state.unsqueeze(0).repeat(repeats, 1))
        if not pieces:
                                                                                          
            pieces = [hidden_states[i, :1]]
        sequence = torch.cat(pieces, dim=0)
        expanded.append(sequence)
        lengths.append(sequence.shape[0])

    max_len = max(lengths)
    output = hidden_states.new_zeros(batch_size, max_len, channels)
    for i, sequence in enumerate(expanded):
        output[i, : sequence.shape[0]] = sequence

    return output, torch.tensor(lengths, device=hidden_states.device, dtype=torch.long)


def fix_len_compatibility(length, num_downsamplings):
                                                                                     
    scale = 2 ** max(int(num_downsamplings), 0)
    if scale <= 1:
        return int(length)
    return ((int(length) + scale - 1) // scale) * scale


def pad_last_dim(x, target_length):
                                                                    
    pad = int(target_length) - int(x.shape[-1])
    if pad <= 0:
        return x
    return F.pad(x, (0, pad))


def pad_time_dim(x, target_length):
                                                            
    pad = int(target_length) - int(x.shape[1])
    if pad <= 0:
        return x
    return F.pad(x.transpose(1, 2), (0, pad)).transpose(1, 2)


def make_group_norm(channels, max_groups=8):
                                                                                       
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormBlock(nn.Module):
                                                                                         

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_mask):
                                                                                       
        mask = text_mask.unsqueeze(-1)
        hidden = self.conv((x * mask).transpose(1, 2)).transpose(1, 2)
        hidden = F.relu(hidden)
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)
        return hidden * mask


class ChannelLayerNorm(nn.Module):
                                                                              

    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
                                                                            
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)


class RelativeSelfAttention(nn.Module):
                                                                                 

    def __init__(self, hidden_size, num_heads, dropout=0.1, max_relative_position=4):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_relative_position = max_relative_position

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relative_bias = nn.Embedding(2 * max_relative_position + 1, num_heads)

    def forward(self, x, text_mask):
                                                                              
        batch_size, time_steps, _ = x.shape

        query = self.query_proj(x).view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(x).view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(x).view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        positions = torch.arange(time_steps, device=x.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.clamp(-self.max_relative_position, self.max_relative_position)
        relative_bias = self.relative_bias(relative_positions + self.max_relative_position)
        scores = scores + relative_bias.permute(2, 0, 1).unsqueeze(0)

                                                                          
        key_padding = ~text_mask.bool()
        scores = scores.masked_fill(key_padding[:, None, None, :], -1e4)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        hidden = torch.matmul(attention, value)
        hidden = hidden.transpose(1, 2).contiguous().view(batch_size, time_steps, self.hidden_size)
        hidden = self.out_proj(hidden)
        return hidden * text_mask.unsqueeze(-1)


class EncoderFeedForward(nn.Module):
                                                                           

    def __init__(self, hidden_size, kernel_size, multiplier, dropout):
        super().__init__()
        inner_size = hidden_size * multiplier
        self.conv1 = nn.Conv1d(hidden_size, inner_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(inner_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_mask):
                                                                                       
        mask = text_mask.unsqueeze(-1)
        hidden = self.conv1((x * mask).transpose(1, 2))
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden).transpose(1, 2)
        hidden = self.dropout(hidden)
        return hidden * mask


class EncoderBlock(nn.Module):
                                                                                  

    def __init__(self, hidden_size, num_heads, attention_dropout, dropout, ffn_kernel_size, ffn_multiplier, max_relative_position):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = RelativeSelfAttention(
            hidden_size,
            num_heads,
            dropout=attention_dropout,
            max_relative_position=max_relative_position,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = EncoderFeedForward(hidden_size, ffn_kernel_size, ffn_multiplier, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_mask):
                                                                                          
        mask = text_mask.unsqueeze(-1)
        attn_out = self.self_attn(self.attn_norm(x), text_mask)
        x = (x + self.dropout(attn_out)) * mask
        ffn_out = self.ffn(self.ffn_norm(x), text_mask)
        x = (x + self.dropout(ffn_out)) * mask
        return x


class TextEncoder(nn.Module):
                                                                                     

    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.prenet = nn.ModuleList(
            [
                ConvNormBlock(
                    config.hidden_size,
                    config.hidden_size,
                    config.encoder_kernel_size,
                    config.encoder_dropout,
                )
                for _ in range(config.encoder_prenet_layers)
            ]
        )
        self.prenet_norm = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    config.hidden_size,
                    config.encoder_num_heads,
                    config.attention_dropout,
                    config.encoder_dropout,
                    config.encoder_ffn_kernel_size,
                    config.encoder_ffn_multiplier,
                    config.encoder_max_relative_position,
                )
                for _ in range(config.encoder_num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, text_ids, text_lengths):
                                                                                  
        text_mask = sequence_mask(text_lengths, text_ids.shape[1]).float()
        mask = text_mask.unsqueeze(-1)

        hidden = self.embedding(text_ids) * mask
        prenet_input = hidden
        for layer in self.prenet:
            hidden = layer(hidden, text_mask)
        hidden = self.prenet_norm(hidden + prenet_input)
        hidden = hidden * mask

        for block in self.blocks:
            hidden = block(hidden, text_mask)

        hidden = self.output_norm(hidden)
        return hidden * mask


class DurationPredictor(nn.Module):
                                                                                           

    def __init__(self, hidden_size, filter_size, num_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_conv = nn.Conv1d(hidden_size, filter_size, kernel_size=kernel_size, padding=kernel_size // 2)
        self.input_norm = ChannelLayerNorm(filter_size)
        self.hidden_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(filter_size, filter_size, kernel_size=kernel_size, padding=kernel_size // 2),
                        "norm": ChannelLayerNorm(filter_size),
                    }
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(filter_size, 1, kernel_size=1)

    def _forward_block(self, hidden, mask, conv, norm):
                                                                          
        hidden = conv(hidden * mask)
        hidden = F.relu(hidden)
        hidden = norm(hidden)
        hidden = self.dropout(hidden)
        return hidden

    def forward(self, hidden_states, text_mask):
                                                                                   
        mask = text_mask.unsqueeze(1)
        hidden = hidden_states.transpose(1, 2)
        hidden = self._forward_block(hidden, mask, self.input_conv, self.input_norm)

        for layer in self.hidden_layers:
            hidden = self._forward_block(hidden, mask, layer["conv"], layer["norm"])

        out = self.proj(hidden * mask).squeeze(1)
        return out * text_mask


class PriorPredictor(nn.Module):
                                                                                            

    def __init__(self, hidden_size, n_mels, num_layers=3, kernel_size=5, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConvNormBlock(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Conv1d(hidden_size, n_mels, kernel_size=1)

    def forward(self, conditioning, frame_mask=None):
                                                                                
        if frame_mask is None:
            frame_mask = conditioning.new_ones(conditioning.shape[0], 1, conditioning.shape[1])
        mask = frame_mask.squeeze(1)
        hidden = conditioning
        for layer in self.layers:
            hidden = hidden + layer(hidden, mask)
        hidden = self.out_norm(hidden) * mask.unsqueeze(-1)
        return self.out_proj(hidden.transpose(1, 2)) * frame_mask


class Mish(nn.Module):
                                                                               

    def forward(self, x):
                                                        
        return x * torch.tanh(F.softplus(x))


class Downsample2d(nn.Module):
                                                                

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
                                                          
        return self.conv(x)


class Upsample2d(nn.Module):
                                                                  

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
                                                              
        return self.conv(x)


class Rezero(nn.Module):
                                                                      

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
                                                                          
        return self.fn(x) * self.g


class Residual(nn.Module):
                                                               

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
                                                           
        return self.fn(x, *args, **kwargs) + x


class Block2d(nn.Module):
                                                           

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, padding=1),
            make_group_norm(dim_out, groups),
            Mish(),
        )

    def forward(self, x, mask):
                                                                                         
        output = self.block(x * mask)
        return output * mask


class ResnetBlock2d(nn.Module):
                                                                                    

    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block2d(dim, dim_out, groups=groups)
        self.block2 = Block2d(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, mask, time_emb, cond_bias=None):
                                                                               
        hidden = self.block1(x, mask)
        hidden = hidden + self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        if cond_bias is not None:
            hidden = hidden + cond_bias
        hidden = self.block2(hidden, mask)
        return hidden + self.res_conv(x * mask)


class LinearAttention2d(nn.Module):
                                                                                

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
                                                                                        
        batch_size, _, height, width = x.shape
        qkv = self.to_qkv(x).view(batch_size, 3, self.heads, self.dim_head, height * width)
        q, k, v = qkv.unbind(dim=1)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(batch_size, self.heads * self.dim_head, height, width)
        return self.to_out(out)


class SinusoidalTimeEmbedding(nn.Module):
                                                                                 

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, scale=1000):
                                                                        
        half_dim = self.dim // 2
        if half_dim == 0:
            return t.unsqueeze(-1)
        exponent = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -exponent)
        angles = scale * t.unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class GradTTSEstimator(nn.Module):
                                                                                 

    def __init__(self, dim, cond_dim, n_mels, dim_mults=(1, 2, 4), groups=8, pe_scale=1000):
        super().__init__()
        self.n_mels = n_mels
        self.pe_scale = pe_scale
        self.num_downsamplings = max(len(dim_mults) - 1, 0)

        self.time_pos_emb = SinusoidalTimeEmbedding(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )

        dims = [2, *[dim * mult for mult in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.down_conds = nn.ModuleList()
        self.down_cond_gains = nn.ParameterList()

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock2d(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                        ResnetBlock2d(dim_out, dim_out, time_emb_dim=dim, groups=groups),
                        Residual(Rezero(LinearAttention2d(dim_out))),
                        Downsample2d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            self.down_conds.append(nn.Conv1d(cond_dim, dim_out, kernel_size=1))
            self.down_cond_gains.append(nn.Parameter(torch.zeros(1)))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock2d(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        self.mid_attn = Residual(Rezero(LinearAttention2d(mid_dim)))
        self.mid_block2 = ResnetBlock2d(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        self.mid_cond = nn.Conv1d(cond_dim, mid_dim, kernel_size=1)
        self.mid_cond_gain = nn.Parameter(torch.zeros(1))

        self.up_conds = nn.ModuleList()
        self.up_cond_gains = nn.ParameterList()
        for dim_in, dim_out in reversed(in_out[1:]):
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock2d(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                        ResnetBlock2d(dim_in, dim_in, time_emb_dim=dim, groups=groups),
                        Residual(Rezero(LinearAttention2d(dim_in))),
                        Upsample2d(dim_in),
                    ]
                )
            )
            self.up_conds.append(nn.Conv1d(cond_dim, dim_in, kernel_size=1))
            self.up_cond_gains.append(nn.Parameter(torch.zeros(1)))

        final_dim = dims[1] if len(dims) > 1 else dim
        self.final_block = Block2d(final_dim, final_dim, groups=groups)
        self.final_conv = nn.Conv2d(final_dim, 1, kernel_size=1)
        self.final_cond = nn.Conv1d(cond_dim, final_dim, kernel_size=1)
        self.final_cond_gain = nn.Parameter(torch.zeros(1))

    def make_cond_bias(self, conditioning, proj, gain, width, dtype):
                                                                                         
        cond_bias = proj(conditioning)
        if cond_bias.shape[-1] != width:
            cond_bias = F.interpolate(cond_bias, size=width, mode="linear", align_corners=False)
        return (gain * cond_bias).unsqueeze(2).to(dtype=dtype)

    def forward(self, x, mask, mu, conditioning, t):
                                                                                  
        time_emb = self.time_pos_emb(t, scale=self.pe_scale)
        time_emb = self.time_mlp(time_emb)
        conditioning = conditioning.transpose(1, 2)

                                                                                   
        hidden = torch.stack([mu, x], dim=1)
        mask_2d = mask.unsqueeze(1)

        skips = []
        masks = [mask_2d]
        for cond_proj, cond_gain, (block1, block2, attn, downsample) in zip(self.down_conds, self.down_cond_gains, self.downs):
            current_mask = masks[-1]
            cond_bias = self.make_cond_bias(conditioning, cond_proj, cond_gain, current_mask.shape[-1], hidden.dtype)
            hidden = block1(hidden, current_mask, time_emb, cond_bias=cond_bias)
            hidden = block2(hidden, current_mask, time_emb, cond_bias=cond_bias)
            hidden = attn(hidden)
            skips.append(hidden)
            hidden = downsample(hidden * current_mask)
                                                                                              
            masks.append(current_mask[:, :, :, ::2])

        masks = masks[:-1]
        current_mask = masks[-1]
        mid_bias = self.make_cond_bias(conditioning, self.mid_cond, self.mid_cond_gain, current_mask.shape[-1], hidden.dtype)
        hidden = self.mid_block1(hidden, current_mask, time_emb, cond_bias=mid_bias)
        hidden = self.mid_attn(hidden)
        hidden = self.mid_block2(hidden, current_mask, time_emb, cond_bias=mid_bias)

        for cond_proj, cond_gain, (block1, block2, attn, upsample) in zip(self.up_conds, self.up_cond_gains, self.ups):
            current_mask = masks.pop()
            cond_bias = self.make_cond_bias(conditioning, cond_proj, cond_gain, current_mask.shape[-1], hidden.dtype)
            hidden = torch.cat([hidden, skips.pop()], dim=1)
            hidden = block1(hidden, current_mask, time_emb, cond_bias=cond_bias)
            hidden = block2(hidden, current_mask, time_emb, cond_bias=cond_bias)
            hidden = attn(hidden)
            hidden = upsample(hidden * current_mask)

        final_bias = self.make_cond_bias(conditioning, self.final_cond, self.final_cond_gain, mask_2d.shape[-1], hidden.dtype)
        hidden = self.final_block(hidden + final_bias, mask_2d)
        output = self.final_conv(hidden * mask_2d)
        return (output * mask_2d).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
                                                                                           
    if cumulative:
        return beta_init * t + 0.5 * (beta_term - beta_init) * (t ** 2)
    return beta_init + (beta_term - beta_init) * t


class DiffusionDecoder(nn.Module):
                                                                                         

    def __init__(
        self,
        hidden_size,
        n_mels,
        num_layers,
        base_channels=64,
        dropout=0.1,
        beta_min=0.05,
        beta_max=20.0,
        pe_scale=1000,
        dim_mults=(1, 2, 4),
        groups=8,
        temperature_mode="inverse",
        x0_loss_weight=0.05,
        sample_clamp_value=4.5,
        prior_layers=3,
        prior_kernel_size=5,
    ):
        super().__init__()
        del num_layers
        self.n_mels = n_mels
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.temperature_mode = temperature_mode
        self.x0_loss_weight = x0_loss_weight
        self.sample_clamp_value = sample_clamp_value
        self.cond_dim = hidden_size
        self.prior_net = PriorPredictor(
            hidden_size,
            n_mels,
            num_layers=prior_layers,
            kernel_size=prior_kernel_size,
            dropout=dropout,
        )
        self.estimator = GradTTSEstimator(
            dim=base_channels,
            cond_dim=hidden_size,
            n_mels=n_mels,
            dim_mults=dim_mults,
            groups=groups,
            pe_scale=pe_scale,
        )
        self.num_downsamplings = self.estimator.num_downsamplings

    def make_prior(self, conditioning, frame_mask=None):
                                                                            
        return self.prior_net(conditioning, frame_mask=frame_mask)

    def sample_terminal(self, mu, temperature):
                                                                                    
        temp = max(float(temperature), 1e-4)
        noise = torch.randn_like(mu)
        if self.temperature_mode == "inverse":
            return mu + noise / temp
        return mu + noise * temp

    def forward_diffusion(self, x0, mask, mu, t):
                                                                                     
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        alpha = torch.exp(-0.5 * cum_noise)
        mean = x0 * alpha + mu * (1.0 - alpha)
        variance = 1.0 - torch.exp(-cum_noise)
        noise = torch.randn_like(x0)
        xt = mean + noise * torch.sqrt(variance)
        return xt * mask, noise * mask

    def score_to_x0(self, xt, mu, score, t):
                                                                                        
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        alpha = torch.exp(-0.5 * cum_noise).clamp_min(1e-4)
        variance = 1.0 - torch.exp(-cum_noise)
        return (xt - mu * (1.0 - alpha) + variance * score) / alpha

    def reverse_diffusion(self, z, mask, mu, conditioning, n_timesteps, stoc=False):
                                                                                  
        if n_timesteps < 1:
            raise ValueError("n_timesteps must be >= 1")
        step_size = 1.0 / float(n_timesteps)
        xt = z * mask

        for step in range(n_timesteps):
                                                                                                          
            t = (1.0 - (step + 0.5) * step_size) * torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
            score = self.estimator(xt, mask, mu, conditioning, t)
            if stoc:
                deterministic = (0.5 * (mu - xt) - score) * noise_t * step_size
                stochastic = torch.randn_like(z) * torch.sqrt(noise_t * step_size)
                delta = deterministic + stochastic
            else:
                delta = (0.5 * (mu - xt) - score) * noise_t * step_size
            xt = (xt - delta) * mask
            if self.sample_clamp_value is not None:
                xt = xt.clamp(-self.sample_clamp_value, self.sample_clamp_value) * mask
        return xt

    @torch.no_grad()
    def sample(self, mu, mask, conditioning, n_timesteps, temperature=1.0, stoc=False):
                                                                                              
        z = self.sample_terminal(mu, temperature) * mask
        return self.reverse_diffusion(z, mask, mu, conditioning, n_timesteps=n_timesteps, stoc=stoc)

    def loss_t(self, x0, mask, mu, conditioning, t):
                                                            
        xt, noise = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        score = self.estimator(xt, mask, mu, conditioning, t)
        noise_estimation = score
        noise_estimation = noise_estimation * torch.sqrt(1.0 - torch.exp(-cum_noise))
        score_loss = torch.sum((noise_estimation + noise) ** 2)
        score_loss = score_loss / (torch.sum(mask).clamp_min(1.0) * self.n_mels)

        x0_pred = self.score_to_x0(xt, mu, score, t) * mask
        clean_loss = torch.abs(x0_pred - x0).mul(mask).sum()
        clean_loss = clean_loss / (torch.sum(mask).clamp_min(1.0) * self.n_mels)
        loss = score_loss + self.x0_loss_weight * clean_loss
        return loss, xt, clean_loss

    def sample_training_timesteps(self, batch_size, dtype, device, offset=1e-5):
                                                                                                 
        half = (batch_size + 1) // 2
        base = torch.rand(half, dtype=dtype, device=device)
        t = torch.cat([base, 1.0 - base], dim=0)[:batch_size]
        return torch.clamp(t, offset, 1.0 - offset)

    def compute_loss(self, x0, mask, mu, conditioning, offset=1e-5):
                                                                                   
        t = self.sample_training_timesteps(x0.shape[0], x0.dtype, x0.device, offset=offset)
        return self.loss_t(x0, mask, mu, conditioning, t)

    def forward(self, z, mask, mu, conditioning, n_timesteps, stoc=False):
                                                                                           
        return self.reverse_diffusion(z, mask, mu, conditioning, n_timesteps=n_timesteps, stoc=stoc)


class CompactSpeechSynth(nn.Module):
                                                                                                   

    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        dim_mults = getattr(config, "diffusion_dim_mults", None)
        if dim_mults is None:
            if config.decoder_layers >= 8:
                dim_mults = (1, 2, 4, 8)
            elif config.decoder_layers >= 6:
                dim_mults = (1, 2, 4)
            else:
                dim_mults = (1, 2)

        self.encoder = TextEncoder(vocab_size, config)
        self.duration_predictor = DurationPredictor(
            config.hidden_size,
            config.duration_predictor_filter_size,
            num_layers=config.duration_predictor_layers,
            kernel_size=config.duration_predictor_kernel_size,
            dropout=config.encoder_dropout,
        )
        self.decoder = DiffusionDecoder(
            config.hidden_size,
            config.n_mels,
            config.decoder_layers,
            base_channels=config.decoder_base_channels,
            dropout=config.decoder_dropout,
            beta_min=getattr(config, "diffusion_beta_min", 0.05),
            beta_max=getattr(config, "diffusion_beta_max", 20.0),
            pe_scale=getattr(config, "diffusion_pe_scale", 1000),
            dim_mults=tuple(dim_mults),
            groups=getattr(config, "diffusion_groups", 8),
            temperature_mode=getattr(config, "diffusion_temperature_mode", "inverse"),
            x0_loss_weight=getattr(config, "diffusion_x0_loss_weight", 0.05),
            sample_clamp_value=getattr(config, "diffusion_sample_clamp_value", 4.5),
            prior_layers=getattr(config, "prior_layers", 3),
            prior_kernel_size=getattr(config, "prior_kernel_size", 5),
        )

    def forward_encoder(self, text_ids, text_lengths):
                                                                                  
        text_mask = sequence_mask(text_lengths, text_ids.shape[1]).float()
        encoder_states = self.encoder(text_ids, text_lengths) * text_mask.unsqueeze(-1)
        detach_duration_input = getattr(self.config, "detach_duration_predictor_input", True)
        duration_input = encoder_states.detach() if detach_duration_input else encoder_states
        log_duration_pred = self.duration_predictor(duration_input, text_mask)
        return encoder_states, log_duration_pred, text_mask

    def prepare_decoder_tensors(self, conditioning, frame_lengths, mel=None):
                                                                                          
        frame_mask = sequence_mask(frame_lengths, conditioning.shape[1]).unsqueeze(1).float()
        prior_mean = self.decoder.make_prior(conditioning, frame_mask=frame_mask) * frame_mask

        target_length = prior_mean.shape[-1]
        if mel is not None:
            target_length = min(target_length, mel.shape[-1])
            mel = mel[:, :, :target_length]

                                                                          
        conditioning = conditioning[:, :target_length]
        prior_mean = prior_mean[:, :, :target_length]
        frame_mask = frame_mask[:, :, :target_length]

        compatible_length = fix_len_compatibility(target_length, self.decoder.num_downsamplings)
        conditioning = pad_time_dim(conditioning, compatible_length)
        prior_mean = pad_last_dim(prior_mean, compatible_length)
        frame_mask = pad_last_dim(frame_mask, compatible_length)
        if mel is not None:
            mel = pad_last_dim(mel, compatible_length)

        return conditioning, mel, prior_mean, frame_mask, target_length

    def crop_training_segment(self, conditioning, mel, prior_mean, frame_mask):
                                                                                                  
        segment_frames = int(getattr(self.config, "decoder_train_segment_frames", 0) or 0)
        if segment_frames <= 0:
            return conditioning, mel, prior_mean, frame_mask

        segment_frames = fix_len_compatibility(segment_frames, self.decoder.num_downsamplings)
        if mel.shape[-1] <= segment_frames:
            return conditioning, mel, prior_mean, frame_mask

        batch_size, channels, _ = mel.shape
        out_conditioning = conditioning.new_zeros(batch_size, segment_frames, conditioning.shape[2])
        out_mel = mel.new_zeros(batch_size, channels, segment_frames)
        out_prior = prior_mean.new_zeros(batch_size, prior_mean.shape[1], segment_frames)
        out_mask = frame_mask.new_zeros(batch_size, 1, segment_frames)
        valid_lengths = frame_mask.squeeze(1).sum(dim=1).long()

        for idx in range(batch_size):
            valid_len = int(valid_lengths[idx].item())
            if valid_len <= 0:
                continue

            copy_len = min(valid_len, segment_frames)
            max_offset = max(valid_len - segment_frames, 0)
                                                                                                    
            start = int(torch.randint(max_offset + 1, (1,), device=mel.device).item()) if max_offset > 0 else 0
            end = start + copy_len

            out_conditioning[idx, :copy_len] = conditioning[idx, start:end]
            out_mel[idx, :, :copy_len] = mel[idx, :, start:end]
            out_prior[idx, :, :copy_len] = prior_mean[idx, :, start:end]
            out_mask[idx, :, :copy_len] = frame_mask[idx, :, start:end]

        return out_conditioning, out_mel, out_prior, out_mask

    def compute_losses(self, batch, noise_levels=None):
                                                                                      
        del noise_levels
        text_ids = batch["text_ids"].to(self.config.device)
        text_lengths = batch["text_lengths"].to(self.config.device)
        durations = batch["durations"].to(self.config.device)
        mel = batch["mel"].to(self.config.device).transpose(1, 2)

        encoder_states, log_duration_pred, text_mask = self.forward_encoder(text_ids, text_lengths)
        duration_target = torch.log1p(durations.float())
        duration_loss = ((log_duration_pred - duration_target) ** 2 * text_mask).sum()
        duration_loss = duration_loss / text_mask.sum().clamp_min(1.0)

        conditioning, frame_lengths = length_regulate(encoder_states, durations)
        conditioning, mel, prior_mean, frame_mask, _ = self.prepare_decoder_tensors(conditioning, frame_lengths, mel=mel)
        conditioning, mel, prior_mean, frame_mask = self.crop_training_segment(conditioning, mel, prior_mean, frame_mask)

        diffusion_loss, _, diffusion_clean_l1 = self.decoder.compute_loss(mel, frame_mask, prior_mean, conditioning)
        norm = frame_mask.sum().clamp_min(1.0) * mel.shape[1]
        prior_l1 = (torch.abs(prior_mean - mel) * frame_mask).sum() / norm
        prior_loss = 0.5 * ((mel - prior_mean) ** 2 + math.log(2 * math.pi))
        prior_loss = (prior_loss * frame_mask).sum() / norm

        total_loss = (
            diffusion_loss
            + getattr(self.config, "prior_loss_weight", 1.0) * prior_loss
            + getattr(self.config, "duration_loss_weight", 0.05) * duration_loss
        )

        return (
            total_loss,
            prior_l1.detach(),
            diffusion_loss.detach(),
            duration_loss.detach(),
            prior_loss.detach(),
        )

    @torch.no_grad()
    def synthesize(self, text, tokenizer, noise_levels, temperature=None, length_scale=None):
                                                                                               
        if temperature is None:
            temperature = getattr(self.config, "inference_temperature", 1.0)
        if length_scale is None:
            length_scale = getattr(self.config, "length_scale", 1.0)

        if noise_levels is None:
            n_timesteps = int(getattr(self.config, "diffusion_steps", 50))
        else:
            n_timesteps = int(len(noise_levels))

        text_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=self.config.device)
        text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.long, device=self.config.device)

        encoder_states, log_duration_pred, _ = self.forward_encoder(text_ids, text_lengths)
        predicted_duration_values = torch.clamp(torch.expm1(log_duration_pred), min=0.0) * float(length_scale)
        durations = torch.clamp(
            torch.round(predicted_duration_values).long(),
            min=0,
            max=self.config.max_predicted_duration,
        )
        empty_rows = durations.sum(dim=1) == 0
        if empty_rows.any():
                                                                      
            durations[empty_rows, 0] = 1

        conditioning, frame_lengths = length_regulate(encoder_states, durations)
        conditioning, _, prior_mean, frame_mask, target_length = self.prepare_decoder_tensors(conditioning, frame_lengths)
        sample = self.decoder.sample(
            prior_mean,
            frame_mask,
            conditioning,
            n_timesteps=n_timesteps,
            temperature=float(temperature),
            stoc=bool(getattr(self.config, "diffusion_stochastic", False)),
        )
        sample = sample[:, :, :target_length]
        prior_mean = prior_mean[:, :, :target_length]

        diagnostics = {
            "text_ids": text_ids.squeeze(0).cpu(),
            "encoder_states": encoder_states.squeeze(0).cpu(),
            "log_duration_pred": log_duration_pred.squeeze(0).cpu(),
            "predicted_duration_values": predicted_duration_values.squeeze(0).cpu(),
            "conditioning": conditioning.squeeze(0).cpu(),
            "token_prior_mel_norm": self.decoder.make_prior(encoder_states).squeeze(0).transpose(0, 1).cpu(),
            "prior_mel_norm": prior_mean.squeeze(0).transpose(0, 1).cpu(),
            "diffusion_steps": n_timesteps,
        }

        return sample.squeeze(0).transpose(0, 1).cpu(), durations.squeeze(0).cpu(), diagnostics
