import logging

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_chans, out_chans):
    return nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False)


def conv1x1(in_chans, out_chans):
    return nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)


class Resblock(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.conv1 = conv3x3(in_dim, dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = conv3x3(dim, dim)
        self.bn2 = nn.BatchNorm2d(dim)
        if dim != in_dim:
            self.res_conv = conv1x1(in_dim, dim)
            self.res_bn = nn.BatchNorm2d(dim)
        else:
            self.res_conv = self.res_bn = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        # residual connection
        if self.res_conv:
            x = self.res_bn(self.res_conv(x))
        y += x
        y = self.relu(y)
        return y


class Network(nn.Module):
    def __init__(self, board_size, input_dim, num_blocks,
                 base_chans, value_chans, policy_chans):
        super().__init__()
        # input upsampling
        self.conv1 = conv3x3(input_dim, base_chans)
        self.bn1 = nn.BatchNorm2d(base_chans)
        # residual blocks
        blocks = [Resblock(base_chans, base_chans)
                  for _ in range(num_blocks)]
        self.resblocks = nn.Sequential(*blocks)
        # value head
        self.value_conv1 = conv1x1(base_chans, value_chans)
        self.value_bn1 = nn.BatchNorm2d(value_chans)
        self.value_fc2 = nn.Linear(value_chans * board_size * board_size, 64)
        self.value_fc3 = nn.Linear(64, 1)
        # policy head
        self.move_conv1 = conv1x1(base_chans, policy_chans)
        self.move_bn1 = nn.BatchNorm2d(policy_chans)
        self.relu = nn.ReLU(inplace=True)

    @property
    def device(self):
        """Get current device of model."""
        return self.conv1.weight.device

    def forward(self, x):
        """
        :param x: Batch of game boards as planes (B, C, H, W)
        """
        # upsample
        x = self.relu(self.bn1(self.conv1(x)))
        # residual blocks
        x = self.resblocks(x)
        # value head
        v = self.relu(self.value_bn1(self.value_conv1(x)))
        v = v.reshape(v.size(0), -1)
        v = self.relu(self.value_fc2(v))
        v = self.value_fc3(v)
        value = torch.tanh(v).squeeze(1)   # (B,)
        # policy head
        p = self.relu(self.move_bn1(self.move_conv1(x)))
        p = p.reshape(p.size(0), -1)          # (B, policy_chans * H * W)
        return value, p

    def load(self, modelpath):
        state = torch.load(modelpath)
        self.load_state_dict(state['model'])
        return state['optimizer']

    def save(self, modelpath, optimizer):
        state = {
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, modelpath)


class HexNetwork(Network):
    """
    Original Azalea Hex network.
    Forward(x, legal_moves) returns dict(value, moves_logprob) over legal moves.
    """
    def __init__(self, board_size=11, num_blocks=6, base_chans=64):
        super().__init__(board_size, input_dim=4, num_blocks=num_blocks,
                         base_chans=base_chans, value_chans=2, policy_chans=4)
        # tile encoder
        self.encoder = nn.Embedding(3, 4)
        # policy head
        self.move_fc = nn.Linear(4 * board_size * board_size,
                                 board_size * board_size)
        nnet = sum(p.nelement() for p in self.parameters())
        nenc = sum(p.nelement() for p in self.encoder.parameters())
        logging.info('Net params: {}'.format(nnet - nenc))
        logging.info('Embedding params: {}'.format(nenc))

    def forward(self, x, legal_moves):
        """
        legal_moves padded with zeros
        :param x: Batch of game boards (batch x height x width, int32)
        :param legal_moves: Batch of legal moves (batch x MAX_MOVES, int32)
        """
        # piece encoder
        x = self.encoder(x.long())  # (B, H, W, 4)
        x = x.permute(0, 3, 1, 2)   # (B, 4, H, W)
        # resnet
        value, p = super().forward(x)
        # policy head
        moves_logit = self.move_fc(p)          # (B, H*W)
        legal_tiles = (legal_moves - 1).clamp(min=0)
        moves_logit = torch.gather(moves_logit, 1, legal_tiles.long())
        # clear padding
        moves_logit.masked_fill_(legal_moves == 0, -99)
        moves_logprob = F.log_softmax(moves_logit, dim=1)
        return dict(value=value, moves_logprob=moves_logprob)


class HexNetworkFull(HexNetwork):
    """
    Wrapper so it matches your MCTS interface.

    Input:
        x: (B, H, W) int board, values 0/1/2.
    Output:
        policy_logits: (B, H*W)
        value:         (B, 1)
    """
    def forward(self, x):
        # x: batch x height x width, ints

        # Encode tiles as in original HexNetwork
        x = self.encoder(x.long())        # (B, H, W, 4)
        x = x.permute(0, 3, 1, 2)        # (B, 4, H, W)

        # Shared trunk
        value, p = Network.forward(self, x)   # value: (B,), p: (B, 4*H*W)

        # Policy logits over all H*W cells
        moves_logit = self.move_fc(p)         # (B, H*W)

        B = moves_logit.size(0)
        policy_logits = moves_logit.reshape(B, -1)
        value = value.reshape(B, 1)

        return policy_logits, value


def load_hex11_pretrained(model_path: str,
                          device: torch.device,
                          board_size: int = 11) -> HexNetworkFull:
    """
    Helper to load Azalea's pretrained Hex model.

    The checkpoint layout (from your errors) looks like:

        raw = {
            "policy": {
                "net": <state_dict or nn.Module>,
                "simulations": ...,
                "search_batch_size": ...,
                ...
            },
            "optimizer": ...
        }

    We want the weights inside raw["policy"]["net"].
    """
    model = HexNetworkFull(board_size=board_size)

    raw = torch.load(model_path, map_location=device)

    # ----- locate the object that actually holds parameters -----
    obj = raw  # default

    if isinstance(raw, dict):
        # First peel "policy" if it exists
        if "policy" in raw:
            pol = raw["policy"]
            # Inside policy, prefer "net" if present
            if isinstance(pol, dict) and "net" in pol:
                obj = pol["net"]
            else:
                obj = pol
        # If no "policy" but direct "net" or "model" etc.
        elif "net" in raw:
            obj = raw["net"]
        elif "model" in raw:
            obj = raw["model"]
        elif "state_dict" in raw:
            obj = raw["state_dict"]
        else:
            obj = raw
    else:
        obj = raw

    # ----- turn obj into a plain state_dict -----
    from torch.nn import Module
    if isinstance(obj, Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise TypeError(f"Unexpected type inside checkpoint: {type(obj)}")

    # ----- finally load into our HexNetworkFull -----
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
