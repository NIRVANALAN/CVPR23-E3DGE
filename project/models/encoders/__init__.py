from .fpn_encoders import HybridGradualStyleEncoder_V2, HybridGradualStyleEncoder
from .restyle_psp_encoders import BackboneEncoderRenderer
from . import e4e_encoders


def set_encoder(full_opts):
    """set encoders

    Args:
        opts (Munch): model options

    Raises:
        Exception: not a valid encoder

    Returns:
        list: [renderer_latent, decoder_latent]. # all in offsets, need to add avg_mean later
    """
    # * todo, replace with registry.
    opts = full_opts.training
    if opts.encoder_type == 'HybridGradualStyleEncoder':  # * for volume renderer G
        encoder = HybridGradualStyleEncoder(50, 'ir_se', 9, opts)
    elif opts.encoder_type == 'VolumeRenderDiscriminator':  # * for volume renderer G
        from project.models.stylesdf_model import VolumeRenderDiscriminatorEncoder
        encoder = VolumeRenderDiscriminatorEncoder(full_opts)
    elif opts.encoder_type == 'Encoder4EditingHybridBaseline':
        encoder = e4e_encoders.Encoder4EditingHybridBaseline(50, 'ir_se', opts)
    elif opts.encoder_type == 'HybridGradualStyleEncoder_V2':  # * for volume renderer G
        encoder = HybridGradualStyleEncoder_V2(50, 'ir_se', -1, opts)
    elif opts.encoder_type == 'OldEncoder':  # * for volume renderer G
        from project.models.stylesdf_model import StyleGANEncoder, VolumeRenderDiscriminatorEncoder, FullEncoder
        renderer_encoder = VolumeRenderDiscriminatorEncoder(full_opts)
        stylegan_encoder = StyleGANEncoder(opts, 10)
        encoder = FullEncoder(renderer_encoder, stylegan_encoder)
    elif opts.encoder_type == 'BackboneEncoderRenderer':  # * for volume renderer G
        # encoder = BackboneEncoderRenderer(50, 'ir_se', 1 if opts.w_space else 9, opts)
        encoder = BackboneEncoderRenderer(50, 'ir_se',
                                          2 if opts.w_space else 9, opts)
    else:
        raise Exception(f'{opts.encoder_type} is not a valid encoders')
    return encoder


def load_old_ckpt(opt):
    renderer_encoder = VolumeRenderDiscriminatorEncoder(
        opt, mean_latent[0]).to(device)
    stylegan_encoder = StyleGANEncoder(opt.training, g_ema.decoder.n_latent,
                                       style_mean_latent).cuda()

    # * load ckpt
    renderer_state_dict = torch.load(opt.model.renderer_encoder_ckpt,
                                     map_location=device)['encoder_state_dict']

    # * remove unmatched K for styleganR
    if opt.model.stylegan_encoder_ckpt != '':
        pt_model_dict = torch.load(opt.model.stylegan_encoder_ckpt,
                                   map_location=device)
        d_model_dict = pt_model_dict['encoder_state_dict']
        print(f'load pretrained from {opt.model.encoder_ckpt}')

    # * move to parallel
    encoder = FullEncoder(opt, renderer_encoder, stylegan_encoder)
