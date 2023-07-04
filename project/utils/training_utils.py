import torch
import random


################### Latent code (Z) sampling util functions ####################
# Sample Z space latent codes for the generator
def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def make_pair_noise(latent_dim, device):
    return torch.randn(1, latent_dim, device=device).repeat_interleave(2,
                                                                       dim=0)


def make_pair_same_noise(batch, latent_dim, device):
    assert (batch % 2) == 0, 'even batch'

    noise = []
    for _ in range(batch // 2):
        noise.append(make_pair_noise(latent_dim, device))

    noise = torch.cat(noise, dim=0)
    return [noise]


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]


# Exponential moving average for generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def _swap_odd_even_index_list(all_feats: list):
    batch_size = all_feats[0].shape[0]
    assert (batch_size % 2) == 0, 'only support even shape[0]'
    swapped_Tensor = torch.zeros_like(Tensor)

    swapped_Tensor[0::2] = Tensor[1::2]
    swapped_Tensor[1::2] = Tensor[0::2]

    return swapped_Tensor


def _swap_odd_even_index(Tensor: torch.Tensor):
    if Tensor is None:
        return Tensor
    batch_size = Tensor.shape[0]
    assert (batch_size % 2) == 0, 'only support even shape[0]'
    swapped_Tensor = torch.zeros_like(Tensor)

    swapped_Tensor[0::2] = Tensor[1::2]
    swapped_Tensor[1::2] = Tensor[0::2]

    return swapped_Tensor


def _duplicate_odd_even_index_view(Tensor: torch.Tensor):
    """avoid inplace swap, maintain grad

    Args:
        Tensor (torch.Tensor): Tensor to be swapped

    Returns:
        Tensor: odd even swapped Tensor
    """
    if Tensor is None:
        return Tensor
    batch_size = Tensor.shape[0]
    assert (batch_size % 2) == 0, 'only support even shape[0]'
    index = torch.zeros_like(Tensor)
    for i in range(0, batch_size, 2):
        index[i] = i
        index[i + 1] = i
        # index[i] = i + 1
        # index[i + 1] = i
    index = index.long()

    swapped_Tensor = torch.gather(Tensor, 0, index)

    return swapped_Tensor


def _swap_odd_even_index_view(Tensor: torch.Tensor):
    """avoid inplace swap, maintain grad

    Args:
        Tensor (torch.Tensor): Tensor to be swapped

    Returns:
        Tensor: odd even swapped Tensor
    """
    if Tensor is None:
        return Tensor
    batch_size = Tensor.shape[0]
    assert (batch_size % 2) == 0, 'only support even shape[0]'
    index = torch.zeros_like(Tensor)
    for i in range(0, batch_size, 2):
        index[i] = i + 1
        index[i + 1] = i
    index = index.long()

    swapped_Tensor = torch.gather(Tensor, 0, index)

    return swapped_Tensor


def merge_loss_dict(d1: dict, d2: dict):
    # for merging cycle loss dicts
    for k, v in d1.items():
        assert k in d2
        if isinstance(v, torch.Tensor):
            if v.ndim == 1:  #
                d1[k] = (d1[k] + d2[k]) / 2
            elif isinstance(k, torch.Tensor):
                d1[k] = torch.cat([d1[k], d2[k]], 0)
        elif isinstance(v, dict):
            d1[k] = merge_loss_dict(d1[k], d2[k])
        else:
            raise NotImplementedError()

    return d1
