import torch
from pdebench_data_handling import FNODatasetSingle, FNODatasetMult, MultiDataset
from anthony_data_handling import PDEDataset2D

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def _get_filename_and_extension(config):
    # Select specific file
    if(config['dataset'] == 'shallow_water'):
        filename = '2D_rdb_NA_NA.h5'
        extension = 'shallow-water'

    elif(config['dataset'] == 'diffusion_reaction'):
        filename = '2D_diff-react_NA_NA.h5'
        extension = 'diffusion-reaction'

    ###
    #  Various Compressible Navier-Stokes Datasets
    ###
    elif(config['dataset'] == 'cfd_rand_0.1_0.01_0.01'):
        filename = '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5'
        extension = 'CFD/2D_Train_Rand'

    elif(config['dataset'] == 'cfd_rand_0.1_0.1_0.1'):
        filename = '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'
        extension = 'CFD/2D_Train_Rand'

    elif(config['dataset'] == 'cfd_rand_0.1_1e-8_1e-8'):
        filename = '2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
        extension = 'CFD/2D_Train_Rand'

    elif(config['dataset'] == 'cfd_rand_1.0_0.01_0.01'):
        filename = '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5'
        extension = 'CFD/2D_Train_Rand'

    elif(config['dataset'] == 'cfd_rand_1.0_0.1_0.1'):
        filename = '2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'
        extension = 'CFD/2D_Train_Rand'

    elif(config['dataset'] == 'cfd_rand_1.0_1e-8_1e-8'):
        filename = '2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
        extension = 'CFD/2D_Train_Rand'

    elif(config['dataset'] == 'cfd_turb_0.1_1e-8_1e-8'):
        filename = '2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
        extension = 'CFD/2D_Train_Turb'

    elif(config['dataset'] == 'cfd_turb_1.0_1e-8_1e-8'):
        filename = '2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
        extension = 'CFD/2D_Train_Turb'


    ###
    #  Various Inompressible Navier-Stokes Datasets
    ###
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-0.h5'):
        filename = 'ns_incom_inhom_2d_512-0.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-1.h5'):
        filename = 'ns_incom_inhom_2d_512-1.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-2.h5'):
        filename = 'ns_incom_inhom_2d_512-2.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-3.h5'):
        filename = 'ns_incom_inhom_2d_512-3.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-4.h5'):
        filename = 'ns_incom_inhom_2d_512-4.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-5.h5'):
        filename = 'ns_incom_inhom_2d_512-5.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-6.h5'):
        filename = 'ns_incom_inhom_2d_512-6.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-7.h5'):
        filename = 'ns_incom_inhom_2d_512-7.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-8.h5'):
        filename = 'ns_incom_inhom_2d_512-8.h5'
        extension = 'NS_incom'
    elif(config['dataset'] == 'ns_incom_inhom_2d_512-9.h5'):
        filename = 'ns_incom_inhom_2d_512-9.h5'
        extension = 'NS_incom'

    # Combine all of the data sets
    elif(config['dataset'] == 'all'):
        #filename = ['2D_rdb_NA_NA.h5', '2D_diff-react_NA_NA.h5']
        #extension = ["/home/cooperlorsung/pdebench_data/2D/{}".format(i) for i in ['shallow-water', 'diffusion-reaction']]

        filename = [
                '2D_rdb_NA_NA.h5',
                '2D_diff-react_NA_NA.h5',
                '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
                '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5',
                '2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5',
                '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5',
                '2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5',
                '2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5',
                '2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5',
                '2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5',
                'heat',
                'burger',
                'adv',
        ]

        paths = [
                'shallow-water',
                'diffusion-reaction',
                'CFD/2D_Train_Rand',
                'CFD/2D_Train_Rand',
                'CFD/2D_Train_Rand',
                'CFD/2D_Train_Rand',
                'CFD/2D_Train_Rand',
                'CFD/2D_Train_Rand',
                'CFD/2D_Train_Turb',
                'CFD/2D_Train_Turb',
                '', '', '',
        ]
        extension = ["/home/cooperlorsung/pdebench_data/2D/{}".format(i) for i in paths]

    elif(any(sub in config['dataset'] for sub in ['heat', 'burger', 'adv'])):
        filename = config['dataset']
        extension = ''

    else:
        raise NotImplementedError("Select shallow_water, diffusion_reaction, or one of the incompressible CFD datasets for now.")

    return filename, extension


def get_single_dataset(config, filename, extension, pretraining=False):
    # Condtiion for PDEBench datasets
    if(filename not in ['heat', 'adv', 'burger']):
        train_data = FNODatasetSingle(
                filename=filename,
                saved_folder="/home/cooperlorsung/pdebench_data/2D/{}".format(extension),
                initial_step=config['initial_step'],
                reduced_resolution=config['reduced_resolution'],
                reduced_resolution_t=config['reduced_resolution_t'],
                reduced_batch=1,
                if_test=False,
                test_ratio=0.2,
                num_samples_max=config['num_data_samples'],

                clip=config['clip'],
                llm=config['llm'],
                bcs=config['bcs'],
                coeff=config['coeff'],
                eq_coeff=config['eq_coeff'],
                sentence=config['sentence'],
                qualitative=config['qualitative'],
                time=config['time'],
                image_size=config['img_size'],

                transfer=config['transfer'],
                
                normalize=config['normalize'],
        )
        val_data = FNODatasetSingle(
                filename=filename,
                saved_folder="/home/cooperlorsung/pdebench_data/2D/{}".format(extension),
                initial_step=1,
                reduced_resolution=config['reduced_resolution'],
                reduced_resolution_t=config['reduced_resolution_t'],
                reduced_batch=1,
                if_test=True,
                test_ratio=0.2,
                num_samples_max=config['num_data_samples'],

                clip=config['clip'],
                llm=config['llm'],
                bcs=config['bcs'],
                coeff=config['coeff'],
                eq_coeff=config['eq_coeff'],
                sentence=config['sentence'],
                qualitative=config['qualitative'],
                time=config['time'],
                image_size=config['img_size'],

                transfer=config['transfer'],

                normalize=config['normalize'],
        )
    else:
        print("\n\nFILENAME: {}\n\n".format(filename))
        subset = 'heat' if('heat' in filename) else 'burger' if('burger' in filename) else 'adv' if('adv' in filename) \
                 else 'heat,adv,buger'
        train_data = PDEDataset2D(
                path="/home/cooperlorsung/NEW_HeatAdvBurgers_9216_downsampled.h5",
                pde="Heat, Burgers, Advection",
                mode="train",
                resolution=[config['sim_time'],64,64],
                augmentation=[],
                augmentation_ratio=0.0,
                shift='None',
                load_all=False,
                device='cuda:0',
                num_samples=int(0.8*config['num_samples']),
                clip=config['clip'],
                llm=config['llm'],
                #downsample=config['downsample'],
                subset=subset,
                #debug=DEBUG,
        )
        val_data = PDEDataset2D(
                path="/home/cooperlorsung/NEW_HeatAdvBurgers_3072_downsampled.h5",
                pde="Heat, Burgers, Advection",
                mode="train",
                resolution=[config['sim_time'],64,64],
                augmentation=[],
                augmentation_ratio=0.0,
                shift='None',
                load_all=True,
                device='cuda:0',
                num_samples=int(0.2*config['num_samples']),
                clip=config['clip'],
                llm=config['llm'],
                #downsample=config['downsample'],
                subset=subset,
                #debug=DEBUG,
        )
        #test_data = PDEDataset2D(
        #        path="/home/cooperlorsung/NEW_HeatAdvBurgers_768_downsampled.h5",
        #        pde="Heat, Burgers, Advection",
        #        mode="valid",
        #        resolution=[config['sim_time'],64,64],
        #        augmentation=[],
        #        augmentation_ratio=0.0,
        #        shift='None',
        #        load_all=True,
        #        device='cuda:0',
        #        num_samples=int(0.2*config['num_samples']),
        #        clip=config['clip'],
        #        llm=config['llm'],
        #        #downsample=config['downsample'],
        #        subset=subset,
        #        #debug=DEBUG,
        #)


    return train_data, val_data, None


def get_multi_dataset(config, filenames, saved_folders, pretraining=False):
    print("\nTRAIN DATA")
    train_data = MultiDataset(
            filenames=filenames,
            saved_folders=saved_folders,
            initial_step=config['initial_step'],
            reduced_resolution=config['reduced_resolution'],
            reduced_resolution_t=config['reduced_resolution_t'],
            reduced_batch=1,
            if_test=False,               
            test_ratio=0.2,
            num_samples_max=config['pretraining_num_samples'] if(pretraining) else config['num_data_samples'],
            sim_time=config['sim_time'],

            clip=config['clip'],
            llm=config['llm'],
            bcs=config['bcs'],
            coeff=config['coeff'],
            eq_coeff=config['eq_coeff'],
            sentence=config['sentence'],   
            qualitative=config['qualitative'],
            time=config['time'],
            image_size=config['img_size'],
            
            normalize=config['normalize'],
    )
    print("\nVAL DATA")
    val_data = MultiDataset(      
            filenames=filenames,
            saved_folders=saved_folders,
            initial_step=config['initial_step'],
            reduced_resolution=config['reduced_resolution'],
            reduced_resolution_t=config['reduced_resolution_t'],
            reduced_batch=1,
            if_test=True,
            test_ratio=0.2,
            num_samples_max=config['pretraining_num_samples'] if(pretraining) else config['num_data_samples'],
            sim_time=config['sim_time'],
                                           
            clip=config['clip'],
            llm=config['llm'],           
            bcs=config['bcs'],
            coeff=config['coeff'],
            eq_coeff=config['eq_coeff'],
            sentence=config['sentence'],
            qualitative=config['qualitative'],
            time=config['time'],
            image_size=config['img_size'],
            
            normalize=config['normalize'],
    )
    print("\nTEST DATA")
    #test_data = MultiDataset(
    #        filenames=filenames,
    #        saved_folders=saved_folders,
    #        initial_step=config['initial_step'],
    #        reduced_resolution=config['reduced_resolution'],
    #        reduced_resolution_t=config['reduced_resolution_t'],
    #        reduced_batch=1,               
    #        if_test=True,
    #        test_ratio=0.2,
    #        num_samples_max=config['pretraining_num_samples'] if(pretraining) else config['num_data_samples'],
    #        sim_time=config['sim_time'],

    #        clip=config['clip'],
    #        llm=config['llm'],
    #        coeff=config['coeff'],
    #        eq_coeff=config['eq_coeff'],
    #        bcs=config['bcs'],
    #        sentence=config['sentence'],
    #        qualitative=config['qualitative'],
    #        time=config['time'],
    #        image_size=config['img_size'],
    #        
    #        normalize=config['normalize'],
    #)

    #return train_data, val_data, test_data
    return train_data, val_data, None


def get_data(config, pretraining=False):
    filename, extension = _get_filename_and_extension(config)

    if(config['dataset'] != 'all'):
        train_data, val_data, test_data = get_single_dataset(config, filename, extension, pretraining)
    else:
        train_data, val_data, test_data = get_multi_dataset(config, filename, extension, pretraining)

    batch_size = config['pretraining_batch_size'] if(pretraining) else config['batch_size']
    print("\nPRETRAINING: {}\n".format(pretraining))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               generator=torch.Generator(device='cuda'),
                                               num_workers=config['num_workers'],
                                               shuffle=True,
                                               #sampler=DistributedSampler(train_data),
                                               )
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                             generator=torch.Generator(device='cuda'),
                                             num_workers=config['num_workers'],
                                             shuffle=True,
                                             #sampler=DistributedSampler(val_data),
                                             )

    # Batch size of 1 makes it easier to evaluate only over relevant channels
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                              generator=torch.Generator(device='cuda'),
                                              num_workers=config['num_workers'],
                                              #shuffle=False,
                                              #sampler=DistributedSampler(val_data),
                                              )

    return train_loader, val_loader, test_loader

