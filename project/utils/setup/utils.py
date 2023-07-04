import torch


def setup_mean_latent(generator, opt, device):
    with torch.no_grad():
        mean_latent = generator.mean_latent(opt.inference.truncation_mean,
                                            device)
        if opt.training.w_space:
            pass
        else:
            mean_latent[0] = mean_latent[0].repeat(
                1, opt.rendering.depth + 1).reshape(
                    1, opt.rendering.depth + 1,
                    mean_latent[0].shape[-1])  # 1, 256 -> 1, 9, 256
            # mean_latent[1] = mean_latent[1].repeat(6, 1).unsqueeze(0)  # 1, 256 -> 1, 6, 512
            mean_latent[1] = mean_latent[1].repeat(10, 1).unsqueeze(
                0)  # 1, 512 -> 1, 10, 512

        surface_mean_latent = mean_latent[0]
    return mean_latent, surface_mean_latent
