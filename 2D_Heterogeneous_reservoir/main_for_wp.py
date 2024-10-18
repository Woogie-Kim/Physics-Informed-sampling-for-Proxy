# ==================================================================================================================== #
# Physics-informed sampling scheme for efficient wellplacement optimization
# Seoul National university
# Global Petroleum Research Center
# MS Thesis program
# Jong-wook Kim
# 2024.08

# Import module
# ==================================================================================================================== #
import gc
import os.path
import numpy as np
import pandas as pd
import warnings
from packages.sampler import *
from packages.optimization import GlobalOpt
from packages.utils import *
from parsing import args
from packages.Bayes_opt import *
warnings.filterwarnings(action='ignore')

# Initialization
# ==================================================================================================================== #
# set permeability
perm = load_matfile(args.perm_mat, 'original')
perm_idx = load_matfile(args.perm_mat, 'selected')
id  = perm_idx[0][0] - 1
# id는 0 ~ 99의 범위를 가짐
# id =
args.permeability = np.array(perm[id])
draw_perm(args.permeability, fname='logperm')
plt.close()
# Setting
tensorboard_setting = {'Use': False,
                            'Filename': f'ECL/Run1',
                            'Tagname': f'Fitness(NPV)'
                            }
sampling_setting = {'Method': args.sampling_method,
                     'Default radius': args.radius_default,
                     'Use Quality': False,
                     'Infeasible ratio': args.ratio_of_infeasible,
                     'Number of Samples': args.num_of_train_sample,
                     'Number of max wells': args.array_of_wells[-1],
                     'Well array': args.array_of_wells,
                     'Permeability': args.permeability,
                     'Quality': make_quality(args.permeability),
                     'Parallel': args.parallel,
                     'Process': args.max_process,
                     'Save': f"wo_{args.num_of_train_sample}_{args.sampling_method}_{id}.pkl"
                    }
optimization_setting = {'Algorithm': args.optimization_algorithm,
                     'Number of particles': args.num_of_particles,
                     'Number of generations': args.num_of_generations,
                     'Proxy': None,
                     'Fine Tune': args.fine_tune,
                     'Use retrain': args.retrain,
                     'Span of retrain': args.span_of_retrain,
                     'Parallel': args.parallel,
                     'Process': args.max_process,
                     'Save':'PSOResult.pkl',
                     'Tensorboard':tensorboard_setting
                        }
Bayesian_opt_setting = {'Batch_size': args.batch_size_bound,
                        'Epoch': args.num_of_epochs_bound,
                        'Lr': args.lr_bound,
                        'Gamma': args.gamma_bound}

proxy_setting = {'Model': args.proxy_name,
                 'Input': args.proxy_input,
                 'Batch_size': args.batch_size,
                 'Epoch': args.num_of_epochs,
                 'Lr': args.lr,
                 'Line search': args.Line_search,
                 'Valid_pick': args.valid_pick,
                 'Scheduler': args.scheduler,
                 'Gamma': args.gamma,
                 'Silent': False,
                 'Device':args.device,
                 'Save': 'saved_model',
                 'Tensorboard':tensorboard_setting,
                 'Bayesian Optimization': Bayesian_opt_setting
                 }
sampling_setting['Sample array'] = setter(sampling_setting)

# Main code for WPO
# ==================================================================================================================== #
def main():
    seed = 0
    description()
    fix_seed(seed)
    if args.use_proxy:
        print('\nWell placement optimization using Proxy model - 2D Heterogeneous Reservoir')
        print('=' * 82)
    # sampling section #################################################################################################
        name = sampling_setting['Method']
        origin = sampling_setting['Number of Samples']
        print('\n1. Sampling Section')
        print('-' * 82)
        if (sampling_setting['Method'] in ['2stage', 'physics']):
            sampling_setting['Method'] = 'uniform'
            sampling_setting['Number of Samples'] = int(origin/2)
            samples_uniform = sampling_section(args, sampling_setting, num_original_samples=origin, original_name=name)
            sampling_setting['Method'] = name
            sampling_setting['Number of Samples'] = origin - sampling_setting['Number of Samples']
            samples = sampling_section(args, sampling_setting, num_original_samples=origin, duplicate=samples_uniform)
        else:
            samples = sampling_section(args, sampling_setting, num_original_samples=origin)

        if args.start_from <= 1:
            visualization_for_sampling_results(samples, name, id)
            visualization_for_quality_maps(samples, sampling_setting, name, id)
    # proxy modeling section ###########################################################################################
        if args.end_to == 1: 
            exit()
        else:
            print('\n2. Proxy Modeling Section')
            print('-' * 82)
            if args.start_from <= 2:
                error_detect_for_hypeopt(args)
                proxy = proxy_modeling_section(args, samples, proxy_setting, name=name, seed=seed)
                visualization_for_proxy_performance(proxy, name, id)
            else: 
                with open(f'./cached/Proxy_{name}_{id}.pkl', 'rb') as f:
                    proxy = pickle.load(f)
    else:
        print('\nWell placement optimization using Eclipse')
        print('=' * 82)
        samples = []; proxy = None; name = 'Eclipse'
    # optimization section #############################################################################################
    if args.end_to ==2: 
        exit()
    else:
        for seed in range(1, args.number_of_optimization + 1):
            if args.start_from > 3:
                recall_optimization_results(name, seed, id)
            else:
                print(f"\n{optimization_setting['Algorithm']} runs --> {seed}/{args.number_of_optimization}")
                WPO = optimization_section(args, sampling_setting, optimization_setting, seed=seed, sample=samples,
                                           proxy=proxy, name=name)
                visualization_for_optmization_results(WPO, name, seed, id)
                del WPO
                gc.collect()


# Codes for each section
# ==================================================================================================================== #
def sampling_section(args, s_setting, num_original_samples, duplicate=None, original_name=None):
    s_setting['Save'] = namer(s_setting, id, num_original_samples)
    s_setting['Sample array'] = setter(s_setting)
    error_detect(s_setting)
    if args.start_from <= 1:
        if (args.direct_load) and (original_name in ['2stage', 'physics']):
            if s_setting['Save'] in os.listdir('./cached'):
                print("There's available cached file for Two-stage sampling. Reuse the 1st samples...")
                samples = Packer_new(s_setting, args=args, permeability=args.permeability, load=True, train_data=False,
                                    duplicate=duplicate)
            else:
                print("There's no available cached file for Two-stage sampling. Generate the new 1st samples...")
                samples = Packer_new(s_setting, args=args, permeability=args.permeability, load=False, train_data=True,
                                     duplicate=duplicate)
        else:
            samples = Packer_new(s_setting, args=args, permeability=args.permeability, load=False, train_data=True,
                                 duplicate=duplicate)
    else:
        samples = Packer_new(s_setting, args=args, permeability=args.permeability, load=True, train_data=False,
                            duplicate=duplicate)
    return samples

def proxy_modeling_section(args, samples, p_setting, name, seed=0):
    preprocess_tof(samples)
    fix_seed(seed)
    if args.use_bayesopt:
        opt = BayesOpt(args, setting=p_setting, samples=samples, seed=seed)
        rst = opt.perform_BayesOpt(init_points=args.num_of_initial_candidates, n_iter=args.num_of_new_observations)
        # opt.return_Param(rst.max, p_setting)
        with open(f'./cached/Proxy_best_Bayesopt.pkl', 'rb') as f:
            proxy = pickle.load(f)

        print(f"R2 Score:{proxy.metric['r2_score'][-1]:.4f}")
        print(f"MAPE: {proxy.metric['MAPE'][-1]:.1f}%")
    else:
        proxy = ProxyModel(args, samples, setting=p_setting)
        proxy.model = proxy.train_model(samples, train_ratio=args.train_ratio, validate_ratio=args.validate_ratio,
                                        saved_dir=proxy.saved_dir, saved_model='saved_model')

    if name:
        with open(f'./cached/Proxy_{name}_{id}.pkl', 'wb') as f:
            pickle.dump(proxy, f)
    return proxy

def optimization_section(args, s_setting, opt_setting, seed=0, sample=[], proxy=None, name=None):
    s_setting = copy(s_setting)
    s_setting = opt_sampling_setter(s_setting, opt_setting)
    if args.use_proxy:
        opt_setting['Proxy'] = proxy
    fix_seed(seed)
    placement_positions = Packer_new(s_setting, args=args, permeability=args.permeability, load=False, save=False,
                                     train_data=False)
    WPO = GlobalOpt(args, placement_positions, args.permeability, setting=opt_setting, sample=sample)
    WPO.iterate(opt_setting['Number of generations'])
    
    if name:
        with open(os.path.join(args.cached_dir, f'./WPO_{name}_{seed}_{id}.pkl'), 'wb') as f:
            pickle.dump(WPO,f)
    
    return WPO

# Visualization for analysis
# ==================================================================================================================== #
def visualization_for_sampling_results(samples, name, id):
    fitness = [p.fit/1e6 for p in samples]
    if (name in ['2stage', 'physics']): 
        split = int(len(fitness)/2)
        NPV_histogram(fitness, seperated=split,bins=np.arange(0, 6.5e2, 0.5e2), is_2stage=True, fname=f'hist_{name}')
    else:    
        NPV_histogram(fitness, bins=np.arange(0, 6.5e2, 0.5e2), is_2stage=False, fname=f'hist_{name}')

def visualization_for_quality_maps(samples, s_setting, name, id):
    split = int(len(samples)/2)
    if name == '2stage':
        # NPV-based quality maps
        _, q_npv, _ = get_CDF(samples[:split], 0.75, view=False, fname=None)
        draw_qualitymap(q_npv, fname=f'{name}_quality')
    elif name == 'physics':
        # Physics-informed quality maps
        q_sl, _,_ = SLquality(args.permeability, samples[:split], s_setting)
        draw_qualitymap(q_sl, fname=f'{name}_quality')

def visualization_for_proxy_performance(proxy, name, id):
    pred = [p[0]/1e6 for p in proxy.predictions]
    true = [p[0]/1e6 for p in proxy.reals]
    get_regression(true, pred,min_=0,max_=550,fname=f'{args.figure_saved_dir}/proxy_{name}_{id}')

def visualization_for_optmization_results(WPO, name, seed, id):
    plt.rcParams['figure.dpi'] = 300
    wellplacement(WPO.algorithm.gbest[-1],filename=f'wellplacement_{name}_{seed}_{id}')
    draw_relative_error(WPO, f'{name}_{seed}')
    draw_graph(WPO.fits_true['gbest'], WPO.fits_proxy['gbest'], f'Graph_{name}_{seed}_{id}')
    
def recall_optimization_results(name, seed, id):
    with open(os.path.join(args.cached_dir, f'./WPO_{name}_{seed}_{id}.pkl'), 'rb') as f:
        WPO = pickle.load(f)
    visualization_for_optmization_results(WPO, name, seed, id)

# Additional
# ==================================================================================================================== #
def description():
    print('\n\nProgram for well placement optimization using CNN-based proxy\n')
    print('Description ' + '='*70)
    print('Title: Physics-Informed Sampling Scheme for Efficient Well Placement Optimization')
    print('Journal: ASME Journal of Energy Resources Technology')
    print("DOI: \033 https://doi.org/10.1115/1.4066103\033")
    print('Developer: Jongwook Kim')
    print('Advisor: John(Jonggeun) Choe')
    print('Contact: kjw97@snu.ac.kr [Jongwook Kim]')
    print('Last revised: 09-Oct-2024')
    print('=' * 82)
# ==================================================================================================================== #
if __name__ == "__main__":
    main()