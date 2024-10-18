import argparse
import torch
import os
# Argument Parsing #####################################################################################################
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

# Main control panel
# ==================================================================================================================== #
# 0. Restart option
# If you want to start or restart from a specific point to the end point, enter the number as shown below.
# {1: sampling, 2: proxy modeling, 3: optimization. 4: results only}
args.start_from = 4
args.end_to = 4

# 1. Sampling
args.sampling_method = 'random'                                     # Sampling Method    ->
args.num_of_train_sample = 10                                       # Number of training samples       default: 2000
args.direct_load = False                                            # Reuse already generated uniform samples

# 2. Proxy modeling
args.proxy_name = 'CNN_3D'                                          # Algorithm name of the proxy      default: 'CNN_3D'
args.use_bayesopt = False                                           # Perform hyperparameter optimization
args.num_of_initial_candidates = 5                                  # Number of initial solutions for Bayes opt
args.num_of_new_observations = 5                                    # Number of new observations for Bayes opt
args.batch_size_bound = (4, 256)                                    # Bound of Bayes_opt -> (lower bound, upper bound)
args.num_of_epochs_bound = (40, 120)
args.lr_bound = (1e-4, 5e-3)
args.gamma_bound = (0.005, 0.01)
args.transform = False                                              # Use data augmentation
args.batch_size = 128                                               # Batch size                       default: 128
args.num_of_epochs = 60                                             # Number of epoch                  default: 60
args.lr = 3e-3                                                      # Learning rate                    default: 3e-3

# 3. Optimization
args.use_proxy = True                                               # Use proxy model | if not, ecl will be used.
args.optimization_algorithm = 'PSO'                                 # Optimization algorithm            default: 'PSO'
args.number_of_optimization = 1                                     # Number of optimization runs       default: 5
args.num_of_generations = 2                                         # Number of generation              default: 100
args.num_of_particles = 2                                           # Swarm size                        default: 40

# Detail parameters for each section
# ==================================================================================================================== #
# Data information  ####################################################################################################
args.filepath = 'data'
args.simulation_directory = 'simulation'
args.save_directory = 'variables'
args.ecl_filename = 'Egg_Model_Eclrun'
args.frs_filename = 'Egg_Model_Frsrun'
args.perm_filename = 'Egg_Model_PERMX'
args.active_filename = 'Egg_model_ACTIVE'
args.position_filename = 'Egg_Model_POSITION'
args.constraint_filename = 'Egg_Model_CONSTRAINT'

# Folder & file generation #############################################################################################
args.train_model_saved_dir_super = './model'
args.train_model_saved_dir = './model/1'
args.train_model_figure_saved_dir = './figure'
args.figure_saved_dir = './fig'
args.cached_dir = './cached'

args.perm_mat = './data/PERM1.mat'
args.active_mat = './data/ACTIVE.mat'
args.perm_ID_mat = './data/PERM1_selected_idx.mat'

# Model information  ###################################################################################################
args.num_of_x = 60
args.num_of_y = 60
args.num_of_z = 2
args.num_of_grid = args.num_of_x * args.num_of_y
args.num_of_min_well = 3                                                    # Number of minimum well       default: 5
args.num_of_max_well = 12                                                   # Number of minimum well       default: 14
args.length_of_x = 36/0.3048 # (m)->(ft), 0.3048 is conversion factor
args.length_of_y = 36/0.3048 # (m)->(ft), 0.3048 is conversion factor
args.length_of_z = 36/0.3048 # (m)->(ft), 0.3048 is conversion factor

# Economic parameters ##################################################################################################
args.discount_rate = 0.1                                                    # Annual discount rate          default: 0.1
args.observed_term = 30                                                     # Discounting Period [month]    default: 30
args.discount_term = 365                                                    # Discounting Period [year]     default: 365
args.oil_price = 60/0.159                                                   # Oil price                     default: 60
args.injection_cost = -5/0.159                                              # Water injection cost          default: -5
args.disposal_cost = -3/0.159                                               # Water disposal cost           default: -3
args.drilling_cost = 2e6                                                    # Well drilling cost            default: 2e6

# Operating & Simulation parameters ####################################################################################
args.production_time = 3600                                                 # Total production period     default: 3600
args.tstep = 30                                                             # Numerical time step         default: 30
args.dstep = 90                                                             # Drilling span               default: 90
args.streamline_time = 30                                                   # Streamline time step        default: 30
args.max_tof = 3600                                                         # Maximum TOF value           default: 3600
args.max_pressure = 3500                                                    # Initial pressure            default: 3500
args.res_oilsat = 0.2                                                       # Residual oil saturation     default: 0.2
args.num_of_rigs = 3                                                        # Number of the drilling rigs default: 3
args.num_of_ensemble = 1                                                    # Number of ensemble          default: 1
args.independent_area = 40                                                  # Independent area            default: 40

# Sampling parameters ##################################################################################################
args.ratio_of_infeasible = 0                                        # Infeasible ratio                      default: 0.3
args.well_type_sampling = {'P': 1, 'I': -1}                         # Type indicator of the wells for uniform sampling
args.well_type = {'P': 1, 'No': 0, 'I': -1}                         # Type indicator of the wells for random sampling
args.available_method = set(['uniform', '2stage', 'physics'])       # Define available sampling methods for sampler

# Proxy modeling parameters ############################################################################################
args.proxy_input = ('Config', 'TOFI', 'TOFP')                       # Input configuration of the proxy
args.train_ratio = 0.7                                              # Training ratio                       default: 0.7
args.validate_ratio = 0.15                                          # Validation ratio                     default: 0.15
args.Line_search = False                                            # Use line search method
args.valid_pick = True                                              # Model selection with low validation error
args.scheduler = 'StepLR'                                           # Scheduling method
args.gamma = 0.1                                                    # hyperparameter of scheduler          default: 0.1
args.num_of_channels = 5                                            # Number of channels for the proxy input
args.stratified = 1                                                # Number of stratified group for data split

# Optimization parameters ##############################################################################################
args.w = 0.729                                                      # Inertia weight                    default: 0.729
args.c1 = 1.494                                                     # Cognitive factor                  default: 1.494
args.c2 = 1.494                                                     # Social factor                     default: 1.494

# Retraining parameters are presented below.
args.retrain = False                                                # Use re-training process
args.fine_tune = False                                              # Use fine-tuning [All of Conv layers are freezed]
args.span_of_retrain = 20                                           # Retraining frequency
args.gen_of_retrain = range(args.span_of_retrain, args.num_of_generations + 1, args.span_of_retrain)

# Computational parameters #############################################################################################
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define the device for training proxy
args.parallel = True                                                       # Parallelization option for numerical sim.
args.max_process = 2                                                        # Number of CPU processor for parallel sim.

# The variables below are fixed value
args.radius_default = True
args.type_fix = False
args.drilling_time_fix = True
args.location_fix = True
args.control_fix = True

args.well_placement_optimization = True
args.well_operation_optimization = False
args.simultaneous_optimization = False
args.well_location_index = None
args.well_type_real = None
args.well_placement_wset = [300, 300, 500, 500] #bar
args.well_operation_wset = [250, 350, 550, 650] #bar

# Automatically calculated parameters [DO NOT CHANGE!]  ################################################################
# Array of the well number
args.array_of_wells = range(args.num_of_min_well, args.num_of_max_well + 1)
rst = [args.num_of_train_sample/ len(args.array_of_wells) for _ in args.array_of_wells]
i = 0
while sum(rst) != args.num_of_train_sample:
    rst[i] += 1
    i += 1
args.array_of_samples = rst

# Caution for train/valid/test split
assert args.validate_ratio != 0, 'validate_ratio should be greater than 0'
assert (1 - args.train_ratio - args.validate_ratio) > 0, '(train_ratio + validate_ratio) should not be 1'
if args.well_operation_optimization and not args.well_placement_optimization:
    # if you only want to optimize well operation conditions, provide well position settings by yourself
    assert args.well_placement_optimization, 'if you only want to optimize well operation conditions, provide ' \
                                             'well position settings by yourself. You must set the default well ' \
                                             'locations for a defined number of wells. ' \
                                             '- well_location_index, well_type_real'

# Folder generation
if not os.path.exists(args.train_model_saved_dir):
    if not os.path.exists(args.train_model_saved_dir_super):
        print('model_saved_dir not exists. Will be generated.')
        os.mkdir(args.train_model_saved_dir_super)
        os.mkdir(args.train_model_saved_dir)
        print('Directory generation finished!')

if not os.path.exists('./cached'):
    if not os.path.exists('./cached'):
        print('cached not exists. Will be generated.')
        os.mkdir('./cached')
        print('Directory generation finished!')

if not os.path.exists('./fig'):
    if not os.path.exists('./fig'):
        print('fig not exists. Will be generated.')
        os.mkdir('./fig')
        print('Directory generation finished!')

# ==================================================================================================================== #


