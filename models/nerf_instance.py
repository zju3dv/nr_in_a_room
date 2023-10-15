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


class NeRF_Instance(nn.Module):
    def __init__(self, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_Instance, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips
        # self.activation = nn.ReLU(True)
        # self.activation = nn.Softplus(True)
        self.activation = nn.LeakyReLU(inplace=True)

        sequential_func = nn.Sequential
        # sequential_func = SublinearSequential

        self.use_checkpoint = False

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = sequential_func(layer, self.activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # instance encoding
        self.instance_embedding_channels = 128

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
        # direction encoding layers
        self.dir_encoding = sequential_func(
            nn.Linear(W + in_channels_dir, W // 2), self.activation
        )

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = sequential_func(nn.Linear(W // 2, 3), nn.Sigmoid())

    def forward_instance_mask(self, inputs):
        """
        Encodes input xyz to instance mask.
        """
        xyz = inputs["xyz_embedded"]
        inst_embedded = inputs["inst_embedded"]
        # embedding_a = inputs['embedding_a']
        input_dir = inputs["input_dir"]

        input_x = torch.cat([xyz, inst_embedded], -1)

        # gather occupancy mask
        N_full = input_x.shape[0]
        # occupancy_mask = xyz[:, 0] != 0
        occupancy_mask = (xyz != 0).any(-1)
        ind_full = torch.arange(N_full)
        # ind_occu = ind_full
        ind_occu = ind_full[occupancy_mask]

        input_x = input_x[ind_occu, ...]
        input_dir = input_dir[ind_occu, ...]
        # embedding_a = embedding_a[ind_occu, ...]
        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            if self.use_checkpoint:
                x_ = checkpoint(getattr(self, f"instance_encoding_{i+1}"), x_)
            else:
                x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma_ = self.instance_sigma(x_)

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)

        # for empty, space, we output a minus sigma
        full_sigma = torch.full(
            (N_full, 1), -1e4, device="cuda", dtype=inst_sigma_.dtype
        )
        full_sigma[ind_occu, :] = inst_sigma_

        full_rgb = torch.full((N_full, 3), 0, device="cuda", dtype=rgb.dtype)
        full_rgb[ind_occu, :] = rgb

        return full_sigma, full_rgb

    def forward(self, x, sigma_only=False, extra_inputs=dict()):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = torch.split(
                x, [self.in_channels_xyz, self.in_channels_dir], dim=-1
            )
        else:
            input_xyz = x

        # occupancy_mask = input_xyz[:, 0] != 0
        occupancy_mask = (input_xyz != 0).any(-1)
        # print(occupancy_mask.sum() / occupancy_mask.numel())

        N_full = input_xyz.shape[0]
        ind_full = torch.arange(N_full)
        ind_occu = ind_full[occupancy_mask]
        # ind_occu = ind_full
        input_xyz = input_xyz[ind_occu, :]
        if not sigma_only:
            input_dir = input_dir[ind_occu, :]

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            if self.use_checkpoint:
                xyz_ = checkpoint(getattr(self, f"xyz_encoding_{i+1}"), xyz_)
            else:
                xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            # for empty, space, we output a minus sigma
            sigma_full = torch.full((N_full, 1), -1e4, device="cuda", dtype=sigma.dtype)
            sigma_full[ind_occu, :] = sigma
            sigma = sigma_full
            xyz_out_full = torch.zeros(
                (N_full, xyz_.shape[1]), device="cuda", dtype=xyz_.dtype
            )
            xyz_out_full[ind_occu, :] = xyz_
            xyz_ = xyz_out_full
            return sigma, xyz_

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        out_full = torch.zeros((N_full, 4), device="cuda", dtype=out.dtype)
        # for empty, space, we output a minus sigma
        out_full[:, -1] = -1e4
        out_full[ind_occu, :] = out
        out = out_full

        xyz_out_full = torch.zeros(
            (N_full, xyz_.shape[1]), device="cuda", dtype=xyz_.dtype
        )
        xyz_out_full[ind_occu, :] = xyz_
        xyz_ = xyz_out_full

        return out, xyz_
