from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from timm.models.layers import DropPath

class DSConv2D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride=1, padding=0, dilation=1,):
        '''2D Depthwise separable convolution.'''
        super().__init__()
        # Depthwise convolution
        self.depthwise = torch.nn.Conv2d(
            ch_in, ch_out, kernel_size=kernel, stride=stride,
            padding=padding, dilation=dilation, groups=ch_in)
        # Linear combination of depthwise convolution
        self.pointwise = torch.nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, ch_in, padding, embed_dim, stride=2, bias=False):
        '''Get patch embeddings from input image.'''
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            ch_in,
            self.embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)  # (b:n_samples, c:embed_dim, h:n_patches**0.5, w:n_patches**0.5)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (b:n_samples, l:n_patches, c:embed_dim)
        x = self.norm(x)
        return x


class CEMSA(nn.Module):
    '''CEMSA Block'''
    def __init__(
            self, dim, image_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
            sr_ratio=1, kernel_size=3, q_stride=1
    ):
        super().__init__()
        self.num_heads = num_heads
        # Tuple (h, w)
        self.img_size = image_size
        head_dim = dim // num_heads
        pad = (kernel_size - q_stride) // 2
        inner_dim = dim
        self.scale = head_dim ** -0.5  # not used
        # Depthwise separable convolution - query matrix
        self.q = DSConv2D(dim, inner_dim, kernel_size, q_stride, pad)
        # Key, values learnable matrices
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # Dropout layer for output of attention heads
        self.attn_drop = nn.Dropout(attn_drop)
        # Projection of attention heads into subspace
        self.proj = nn.Linear(dim, dim)
        # Dropout layer for projection of attention heads
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        # Use Spatial-Reduction Attention
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim, bias=False)
            self.sr_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        b, n, _, num_heads = *x.shape, self.num_heads
        xq = rearrange(
            x, 'b (l w) n -> b n l w',
            l=self.img_size[0], w=self.img_size[1], d=self.img_size[2]
        )
        q = self.q(xq)
        q = rearrange(q, 'b (h d) l w k -> b h (l w k) d', h=num_heads)

        # Use Spatial-Reduction Attention
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, self.img_size[0], self.img_size[1], self.img_size[2])
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MSA(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias=False, attn_p=0., proj_p=0.):
        '''Multi-head Attention mechanism (scaled dot product attention).'''
        super().__init__()
        # Number of transformer heads
        self.n_heads = n_heads
        # Patch embeddings dimension
        self.dim = dim
        # Self-attention computation attention
        self.head_dim = dim // n_heads
        # Divisor of scaled dot product attention
        self.scale = self.head_dim ** -0.5
        # Linear projection of embeddings with the query, key and value matrices
        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=qkv_bias)
        # Dropout probability for the attention weights
        self.attn_drop = nn.Dropout(attn_p)
        # Linear projection of attention heads
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        # Dropout probability for the projection layer
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        # Check patch embedding dimensions match
        if dim != self.dim:
            raise ValueError
        qkv = self.qkv(x).reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)  
        qkv = rearrange(qkv,
            'n_samples n_tokens qkv n_heads head_dim -> qkv n_samples n_heads n_tokens head_dim')
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples,n_heads,head_dim,n_patches+1)
        # 1. Dot product
        dp = (q @ k_t) * self.scale  # (n_samples,n_heads,n_patches+1,n_patches+1)
        # 2. Softmax
        attn = dp.softmax(dim=-1)  # (n_samples,n_heads,n_patches+1,n_patches+1)
        # Dropout layer for attention output
        attn = self.attn_drop(attn)
        # Average attention weights over heads
        weighted_avg = attn @ v  # (n_samples,n_heads,n_patches+1,head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)
        # (n_samples,n_patches+1,n_heads,head_dim)
        weighted_avg = rearrange(
            weighted_avg, 'n_samples n_patches n_heads head_dim -> n_samples n_patches (n_heads head_dim)'
        )  # (n_samples,n_patches+1,dim)
        # Linear projection of attention heads
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p):
        '''Multilayer perceptron head.'''
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)  # (n_samples, n_patches+1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches+1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches+1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches+1, out_features)
        x = self.drop(x)  # (n_samples, n_patches+1, out_features)
        return x

class Transformer(nn.Module):
    def __init__(self,
                 dim, image_size, n_heads, sr_ratio, drop_path_ratio=0., mlp_ratio=4.0,
                 qkv_bias=True, p=0., attn_p=0., proj_drop=0
                 ):
        '''
        Transformer block.
        '''
        super().__init__()
        # Stochastic depth
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        # Multi-head self-attention (original implementation)
        self.attn = MSA(
            dim, image_size, num_heads=n_heads, qkv_bias=qkv_bias,
            sr_ratio=sr_ratio, proj_drop=proj_drop, attn_drop=attn_p)
        # Layer normalization
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # Ratio of mlp hidden dim to embedding dim.
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLPHead(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            p=p)

    def forward(self, x):
        x = x + self.drop_path(
            self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.mlp(self.norm2(x))
        )
        return x


def pair(t):
    """
    Get each dimension numbers.
    :param t: tuple
        A image shape '(depth, height, width)' TODO: important, decide if RGB or preprocess 
    :return: int, int ,int
        Three elements represent 'depth','height','width'.
    """
    return t if isinstance(t, tuple) else (t, t, t)


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size, patch_size, embed_dim, sr_ratio,
                 depth, n_heads, drop_path_ratio=0., mlp_ratio=4., qkv_bias=True, p=0., att_p=0.
                 ):
        '''
        Simplified implementation of the Vision transformer.

        :param depth: Total number of transformer blocks.
        '''
        super().__init__()
        # Input image size
        self.img_size = img_size
        # Size of image patches
        self.patch_size = patch_size
        # Get input image and patches height, width and depth (color channels)
        img_depth, img_height, img_width = pair(img_size)
        patch_depth, patch_height, patch_width = pair(patch_size)
        # Dimension of the patch embedding space
        self.embed_dim = embed_dim
        # Total number of image patches TODO: why splitting on depth?
        self.n_patches = img_height * img_width
        self.pos_drop = nn.Dropout(p=p)
        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Transformer(
                    dim=embed_dim,
                    image_size=self.img_size,
                    n_heads=n_heads,
                    sr_ratio=sr_ratio,
                    drop_path_ratio=drop_path_ratio,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=att_p,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class SpatialTransform(nn.Module):
    """
    Spatial transformer block.
    """
    def __init__(self, size):
        super().__init__()
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow, mode='bilinear'):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """
    def __init__(self, inshape, nsteps):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransform(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        # vec = torch.exp(vec)
        return vec


class PatchExpanding(nn.Module):
    def __init__(self, dim, input_shape, scale_factor=2, bias=False):
        """
        Expand operation in decoder. 2**3
        :param dim: input token channels for expanding.
        :param scale_factor: the expanding scale for token channels.
        """
        super().__init__()
        self.dim = dim
        self.input_shape = input_shape
        self.expander = nn.Sequential(
            # Expand size of feature map by factor of 2
            nn.Linear(self.dim, scale_factor * self.dim, bias=bias),
            Rearrange('b (D H W) c -> b D H W c', D=self.input_shape[0], H=self.input_shape[1],W=self.input_shape[2]),
            Rearrange('b D H W (h d w c) -> b (D d) (H h) (W w) c', d=2,h=2, w=2, c=self.dim),
            Rearrange('b D H W c -> b (D H W) c'),
            # Expand dimension by factor of 2
            nn.Linear(self.dim, self.dim // 2)
        )
        self.norm = nn.LayerNorm(self.dim // 2, eps=1e-6)

    def forward(self, x):
        x = self.norm(self.expander(x))
        return x


class SkipConnection(nn.Module):
    def __init__(self, dim, input_shape, bias=False):
        """
        :param dim: each concatenating tokens dim.
        """
        super().__init__()
        self.input_shape = input_shape
        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = rearrange(x, 'b (D H W) c -> b c D H W', D=self.input_shape[0], H=self.input_shape[1],
                      W=self.input_shape[2])
        y = rearrange(y, 'b (D H W) c -> b c D H W', D=self.input_shape[0], H=self.input_shape[1],
                      W=self.input_shape[2])
        x = torch.cat([x, y], 1)
        x = self.fusion(x)
        return x


class RegTran(nn.Module):
    def __init__(self, feature_shape, base_channel, down_ratio, vit_depth, patch_size, n_heads, sr_ratio, learning_mode,
                 emb_bias=False, qkv_bias=False):
        """
        Vit for registration with some progress.
        :param feature_shape: tuple
            Feature maps shape where the feature maps firstly input the encoder vit.
        :param C: int
            The constant represent the dimension output from the Vit.
        :param vit_depth: int
            Each level the transformer depth.
        :param patch_size: int
            Patch size through the whole model.
        :param n_heads: int
            Number of heads.
        """
        super().__init__()

        self.stride = 2
        self.learning_mode=learning_mode

        assert list(patch_size) == [patch_size[0]] * len(patch_size), 'Patch size must be squared.'

        self.pad = ((patch_size[0] - self.stride) // 2) + 1
        # Level 0: 1/1*origin_shape
        self.encoder_conv_1 = nn.Sequential(
            nn.Conv2d(2, base_channel // 2, 3, 1, 1, bias=False),
            nn.Conv2d(base_channel // 2, base_channel // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(base_channel // 2, eps=1e-6)
        )

        # Level 1: 1/2*origin_shape
        self.encoder_conv_2 = nn.Sequential(
            nn.Conv2d(base_channel // 2, base_channel, 3, 2, 1, bias=False),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(base_channel, eps=1e-6)
        )  # shape: [1,16,56,56]

        # Level 2: 1/4*origin_shape
        self.patch_emb_1 = PatchEmbed(
            patch_size=patch_size, in_chans=base_channel, padding=self.pad,
            embed_dim=base_channel * 2
        )  # [1,28*28,C]
        self.img_size = tuple(shape // down_ratio[0] for shape in feature_shape)

        self.encoder_vit_1 = VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            embed_dim=base_channel * 2,
            sr_ratio=sr_ratio[0],
            # depth=vit_depth,
            depth=4,
            n_heads=n_heads[0]
        )  # [1,28*28,1*C]

        # Level 3: 1/8*origin_shape
        self.patch_emb_2 = nn.Sequential(
            Rearrange('b (D H W) c -> b c D H W', D=self.img_size[0], H=self.img_size[1], W=self.img_size[2]),
            PatchEmbed(patch_size=patch_size, in_chans=base_channel * 2,
                       padding=self.pad, embed_dim=base_channel * int(2 ** 2)),
        )  # [1,14*14,1*C]
        self.img_size = tuple(shape // down_ratio[1] for shape in feature_shape)

        self.encoder_vit_2 = VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            embed_dim=base_channel * int(2 ** 2),
            sr_ratio=sr_ratio[1],
            # depth=vit_depth,
            depth=2,
            n_heads=n_heads[1]
        )  # [1,17*17,2*C]

        # Level 4: 1/16*origin_shape, bottle_neck
        self.patch_emb_3 = nn.Sequential(
            Rearrange('b (D H W) c -> b c D H W', D=self.img_size[0], H=self.img_size[1], W=self.img_size[2]),
            PatchEmbed(patch_size=patch_size, in_chans=base_channel * int(2 ** 2),
                       padding=self.pad, embed_dim=base_channel * int(2 ** 3)),
        )  # [1,7*7,2*C]
        self.img_size = tuple(shape // down_ratio[2] for shape in feature_shape)

        self.encoder_vit_3 = VisionTransformer(
            img_size=self.img_size,
            patch_size=patch_size,
            embed_dim=base_channel * int(2 ** 3),
            sr_ratio=sr_ratio[2],
            # depth=vit_depth,
            depth=2,
            n_heads=n_heads[2]
        )  # [1,7*7,4*C]

        """----------------------Decoder-----------------------"""

        self.patch_exp_1 = PatchExpanding(
            dim=base_channel * int(2 ** 3), input_shape=self.encoder_vit_3.img_size
        )  # [1,12*14*12,2*C]
        # self.cat_fuse_1 = SkipConnection(self.encoder_vit_2.embed_dim)
        # output_shape:[1,2C,12,14,12]
        self.cat_fuse_1 = SkipConnection(self.encoder_vit_2.embed_dim, self.encoder_vit_2.img_size)
        self.patch_emb_dec_1 = nn.Sequential(
            PatchEmbed(3, self.encoder_vit_2.embed_dim, 1, self.encoder_vit_2.embed_dim, 1)
        )  # output_shape:[1,12*14*12,2C]
        self.decoder_vit_1 = VisionTransformer(
            img_size=self.encoder_vit_2.img_size,
            patch_size=patch_size,
            embed_dim=self.encoder_vit_2.embed_dim,
            sr_ratio=sr_ratio[1],
            depth=2,
            n_heads=n_heads[1]
        )  # [1,12*14*12,2*C]

        self.patch_exp_2 = PatchExpanding(
            dim=base_channel * int(2 ** 2), input_shape=self.decoder_vit_1.img_size
        )  # [1,24*28*24,4*C]
        # self.cat_fuse_2 = SkipConnection(self.encoder_vit_1.embed_dim)
        # output_shape:[1,24,28,24,4C]
        self.cat_fuse_2 = SkipConnection(self.encoder_vit_1.embed_dim, self.encoder_vit_1.img_size)
        self.patch_emb_dec_2 = nn.Sequential(
            PatchEmbed(3, self.encoder_vit_1.embed_dim, 1, self.encoder_vit_1.embed_dim, 1)
        )  # output_shape:[1,24*28*24,4C]
        self.decoder_vit_2 = VisionTransformer(
            img_size=self.encoder_vit_1.img_size,
            patch_size=patch_size,
            embed_dim=self.encoder_vit_1.embed_dim,
            sr_ratio=sr_ratio[0],
            depth=2,
            n_heads=n_heads[0]
        )  # [1,24*28*24,4*C]
        self.decoder_conv_1 = nn.Sequential(
            Rearrange(
                'b (D H W) c -> b c D H W',
                D=self.encoder_vit_1.img_size[0],
                H=self.encoder_vit_1.img_size[1],
                W=self.encoder_vit_1.img_size[2],
            )
        )  # [1,32,24,28,24]

        self.upsample_1 = nn.ConvTranspose3d(
            in_channels=base_channel * 2, out_channels=base_channel * 1, kernel_size=2,
            stride=2, bias=True
        )
        self.decoder_conv_2 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 1, 3, 1, 1, bias=True),
            nn.Conv2d(base_channel * 1, base_channel * 1, 3, 1, 1, bias=True),
            nn.InstanceNorm3d(base_channel * 1, eps=1e-6)
        )
        self.upsample_2 = nn.ConvTranspose3d(
            base_channel * 1, base_channel // 2, 2, 2, bias=True
        )
        self.decoder_conv_3 = nn.Conv2d(base_channel, base_channel // 2, 3, 1, 1, bias=True)

        self.flow1 = nn.Sequential(
            nn.Conv2d(base_channel // 2, base_channel // 2, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(base_channel // 2, 3, 3, 1, 1, bias=True)
        )

        # diffeomorphic learning
        if self.learning_mode == 'diffeomorphic':
            self.Vec = VecInt([96, 112, 96], 7)
        else:
            pass

    def forward(self, moving, atlas):
        x_conv1 = self.encoder_conv_1(torch.cat([moving, atlas], 1))
        x_conv2 = self.encoder_conv_2(x_conv1)

        x = self.patch_emb_1(x_conv2)
        x_enc_vit1 = self.encoder_vit_1(x)

        x = self.patch_emb_2(x_enc_vit1)
        x_enc_vit2 = self.encoder_vit_2(x)

        x = self.patch_emb_3(x_enc_vit2)
        x_enc_vit3 = self.encoder_vit_3(x)
        x = self.patch_exp_1(x_enc_vit3)
        x = self.cat_fuse_1(x_enc_vit2, x)
        x = self.patch_emb_dec_1(x)
        x = self.decoder_vit_1(x)

        x = self.patch_exp_2(x)
        x = self.cat_fuse_2(x_enc_vit1, x)
        x = self.patch_emb_dec_2(x)
        x = self.decoder_vit_2(x)
        x = self.decoder_conv_1(x)  # Only reshape operation

        x = self.upsample_1(x)
        x = self.decoder_conv_2(torch.cat([x_conv2, x], 1))

        x = self.upsample_2(x)
        x = self.decoder_conv_3(torch.cat([x_conv1, x], 1))

        flow = self.flow1(x)

        if self.learning_mode == 'diffeomorphic':
            flow = self.Vec(flow)
        else:
            pass

        return flow
