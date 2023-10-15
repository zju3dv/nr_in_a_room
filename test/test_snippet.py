import torch
import numpy as np
import ipdb
from kornia import create_meshgrid

grid_size = 30
grid = create_meshgrid(
    grid_size, grid_size, normalized_coordinates=True, device="cuda"
)  # [grid_size, grid_size, 2]

ipdb.set_trace()
