import torch
from models.fno import FNO2d
from models.pitt import StandardPhysicsInformedTokenTransformer2D, LLMPITT2D
from models.dpot import DPOTNet, LLMDPOTNet
from models.lucidrains_vit import ViT, CLIPViT, LLMCLIPViT

def get_neural_operator(model_name, config, data_channels):
    device = config['device']
    if(model_name == "fno"):
        # Correct for number of channels calculated earlier
        model = FNO2d(data_channels, config['modes1'], config['modes2'], config['width'], initial_step=config['initial_step'],
                      dropout=config['dropout'])
    elif(model_name == "unet"):
        model = UNet2d(in_channels=config['initial_step'], init_features=config['init_features'], dropout=config['dropout'])
    elif(model_name == "oformer"):
        encoder = SpatialTemporalEncoder2D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], heads=config['heads'])
                            #, dropout=config['dropout'],
                            #res=config['enc_res'])
        decoder = PointWiseDecoder2D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     propagator_depth=config['decoder_depth'], scale=config['scale'], out_steps=1)
        model = OFormer2D(encoder, decoder, num_x=config['num_x'], num_y=config['num_y'])
    elif(model_name == 'deeponet'):
        model = DeepONet2D(layer_sizes_branch=config['branch_net'], layer_sizes_trunk=config['trunk_net'],
                                activation=config['activation'], kernel_initializer=config['kernel_initializer'])

    model.to(device)
    return model


def get_transformer(model_name, config):
    device = config['device']

    # Get channels based on data set (only applicable to PITT for now)
    data_channels = 1 if(config['dataset'] == 'shallow_water') else 2 if(config['dataset'] == 'diffusion_reaction') else 4
    data_channels = 4 #TODO: one day I'll have this all sorted out...

    # Channels based on initial steps
    channels = config['initial_step']

    # Multiply by number of target variables
    channels *= 1 if(config['dataset'] == 'shallow_water') else 2 if(config['dataset'] == 'diffusion_reaction') else 4

    # Add 2 for grid
    channels += 2

    # Add more for coefficients
    channels += 5 if(config['coeff']) else 0

    # Add time channel if doing arbitraty_step training
    channels += 1 if(config['train_style'] == 'arbitrary_step') else 0

    # Out channels is number of target variables, only doing single step prediction for now
    out_channels = 1 if(config['dataset'] == 'shallow_water') else 2 if(config['dataset'] == 'diffusion_reaction') else 4

    # Create the transformer model.
    if(config['sentence']):
        print("USING SENTENCE CLIP VISION TRANSFORMER WITH: {}\n".format(config['llm']))
        transformer = LLMCLIPViT(
                   image_size=config['img_size'],
                   patch_size=config['patch_size'],
                   dim=config['dim'],
                   depth=config['depth'],
                   heads=config['heads'],
                   mlp_dim=config['mlp_dim'],
                   pool=config['pool'],
                   channels=channels,
                   out_channels=out_channels,
                   dim_head=config['dim_head'],
                   dropout=config['dropout'],
                   emb_dropout=config['emb_dropout'],
                   llm=config['llm'],
        ).to(device)

    elif(model_name == 'vit'):
        print("USING STANDARD VISION TRANSFORMER WITH: {}\n".format(config['llm']))
        transformer = ViT(
                   image_size=config['img_size'],
                   patch_size=config['patch_size'],
                   dim=config['dim'],
                   depth=config['depth'],
                   heads=config['heads'],
                   mlp_dim=config['mlp_dim'],
                   pool=config['pool'],
                   channels=channels,
                   out_channels=out_channels,
                   dim_head=config['dim_head'],
                   dropout=config['dropout'],
                   emb_dropout=config['emb_dropout'],
        ).to(device)

    elif(model_name == 'clipvit'):
        print("USING CLIP VISION TRANSFORMER WITH: {}\n".format(config['llm']))
        transformer = CLIPViT(
                   image_size=config['img_size'],
                   patch_size=config['patch_size'],
                   dim=config['dim'],
                   depth=config['depth'],
                   heads=config['heads'],
                   mlp_dim=config['mlp_dim'],
                   pool=config['pool'],
                   channels=channels,
                   out_channels=out_channels,
                   dim_head=config['dim_head'],
                   dropout=config['dropout'],
                   emb_dropout=config['emb_dropout'],
                   llm=config['llm'],
        ).to(device)

    elif(model_name == 'transolver'):
        transformer = EmbeddingTransolver(
                space_dim=channels-2,      # Spatial - grid dimensions
                fun_dim=2,                 # grid dimension
                out_dim = out_channels,    # Number of output channels varies based on dataset
                H=config['img_size'],
                W=config['img_size'],
                llm=config['llm'],
        ).to(device)

    elif(model_name == "pitt"):
        print("\n USING STANDARD EMBEDDING")
        print("DATA CHANNELS: {}".format(data_channels))
        neural_operator = get_neural_operator(config['neural_operator'], config, data_channels)
        #transformer = StandardPhysicsInformedTokenTransformer2D(
        transformer = LLMPITT2D(
                               input_dim=config['input_dim'],
                               hidden_dim=config['hidden'],
                               num_layers=config['layers'],
                               num_heads=config['heads'],
                               img_size=config['img_size'],
                               neural_operator=neural_operator,
                               dropout=config['dropout'],
                               data_channels=data_channels
        ).to(device=device)

    elif(model_name == "dpot"):
        print("\nUSING DPOT")
        transformer = DPOTNet(
                img_size=config['img_size'],
                patch_size=config['patch_size'],
                mixing_type=config['mixing_type'],
                in_channels=config['in_channels'],
                out_channels=config['out_channels'],
                in_timesteps=config['initial_step'],
                out_timesteps=config['T_bundle'],
                n_blocks=config['n_blocks'],
                embed_dim=config['width'],
                out_layer_dim=config['out_layer_dim'],
                depth=config['depth'],
                modes=config['modes'],
                mlp_ratio=config['mlp_ratio'],
                n_cls=config['n_cls'],
                normalize=config['normalize'],
                act=config['act'],
                time_agg=config['time_agg']
        ).to(device)

    elif(model_name == "llmdpot"):
        print("\nUSING DPOT")
        transformer = LLMDPOTNet(
                img_size=config['img_size'],
                patch_size=config['patch_size'],
                mixing_type=config['mixing_type'],
                in_channels=config['in_channels'],
                out_channels=config['out_channels'],
                in_timesteps=config['initial_step'],
                out_timesteps=config['T_bundle'],
                n_blocks=config['n_blocks'],
                embed_dim=config['width'],
                out_layer_dim=config['out_layer_dim'],
                depth=config['depth'],
                modes=config['modes'],
                mlp_ratio=config['mlp_ratio'],
                n_cls=config['n_cls'],
                normalize=config['normalize'],
                act=config['act'],
                time_agg=config['time_agg'],
                llm=config['llm']
        ).to(device)

    else:
        raise ValueError("Invalid model choice.")
    return transformer

