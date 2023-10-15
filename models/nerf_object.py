import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class NeRF_Object(nn.Module):
    def __init__(
        self, conf, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4]
    ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_Object, self).__init__()
        self.conf = conf
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        # self.activation = nn.ReLU(True)
        self.activation = nn.ReLU6(True)
        # self.activation = nn.Softplus(True)
        # self.activation = nn.LeakyReLU(inplace=True)

        sequential_func = nn.Sequential
        # sequential_func = SublinearSequential

        self.use_checkpoint = False

        # instance encoding
        self.instance_embedding_channels = self.conf["N_obj_embedding"]

        self.inst_channel_in = in_channels_xyz + self.instance_embedding_channels
        # instance mask encoding layers
        # self.inst_D = D // 2
        # self.inst_W = W // 2
        # self.inst_skips = [x//2 for x in skips]

        self.inst_D = D
        self.inst_W = W
        self.inst_skips = skips
        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = sequential_func(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        self.instance_encoding_final = sequential_func(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = sequential_func(
            nn.Linear(self.inst_W + in_channels_dir, self.inst_W // 2), self.activation
        )
        self.inst_rgb = sequential_func(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())

    def forward_instance_mask(self, inputs):
        """
        Encodes input xyz to instance mask.
        """
        xyz = inputs["xyz_embedded"]
        inst_embedded = inputs["inst_embedded"]
        input_dir = inputs.get("input_dir", None)

        input_x = torch.cat([xyz, inst_embedded], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            if self.use_checkpoint:
                x_ = checkpoint(getattr(self, f"instance_encoding_{i+1}"), x_)
            else:
                x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma_ = self.instance_sigma(x_)

        if input_dir is None:
            return inst_sigma_

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)

        return inst_sigma_, rgb

    def forward_instance_mask_skip_empty(self, inputs):
        """
        Encodes input xyz to instance mask.
        """
        xyz = inputs["xyz_embedded"]
        inst_embedded = inputs["inst_embedded"]
        input_dir = inputs.get("input_dir", None)

        input_x = torch.cat([xyz, inst_embedded], -1)

        # gather occupancy mask
        N_full = input_x.shape[0]
        # occupancy_mask = xyz[:, 0] != 0
        occupancy_mask = (xyz != 0).any(-1)
        ind_full = torch.arange(N_full)
        # ind_occu = ind_full
        ind_occu = ind_full[occupancy_mask]

        input_x = input_x[ind_occu, ...]
        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            if self.use_checkpoint:
                x_ = checkpoint(getattr(self, f"instance_encoding_{i+1}"), x_)
            else:
                x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma_ = self.instance_sigma(x_)

        # for empty, space, we output a minus sigma
        full_sigma = torch.full(
            (N_full, 1), -1e4, device="cuda", dtype=inst_sigma_.dtype
        )
        full_sigma[ind_occu, :] = inst_sigma_

        if input_dir is None:
            return full_sigma

        x_final = self.instance_encoding_final(x_)
        input_dir = input_dir[ind_occu, ...]
        dir_encoding_input = torch.cat([x_final, input_dir], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)

        full_rgb = torch.full((N_full, 3), 0, device="cuda", dtype=rgb.dtype)
        full_rgb[ind_occu, :] = rgb

        return full_sigma, full_rgb
