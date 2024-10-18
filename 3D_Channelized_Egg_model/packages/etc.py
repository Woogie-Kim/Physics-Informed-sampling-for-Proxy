'''
혹시나 몰라 분석용 코드 백업
'''

r_no = [s.fit for s in samples_random]
u_no = [s.fit for s in samples_uniform]
r_q = [s.fit for s in samples_random_quality]
u_q = [s.fit for s in samples_uniform_quality]
u_lq = [s.fit for s in samples_uniform_quality_log]
samples_results = pd.DataFrame()
counts_results = pd.DataFrame()
samples_results['Random'] = r_no
samples_results['Uniform'] = u_no
samples_results['Random Quality'] = r_q
samples_results['Uniform Quality'] = u_q
samples_results['Uniform log Quality'] = u_lq
for key in samples_results.keys():
    counts_results[key], bins = np.histogram(samples_results[key], bins=np.arange(-1e8, 5.5e8, 5e7))

########################################################################################################################
plt.style.use('default')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots()
ax.hist(bins[:-1], bins=bins,weights= counts_results['Random'],alpha=0.5, label='Random')
ax.hist(bins[:-1], bins=bins,weights= counts_results['Random Quality'],alpha=0.5, label='Random Quality')
ax.set_xlabel('Net Present Value, $',fontname=typo)
ax.set_ylabel('Frequency',fontname=typo)
plt.legend()
plt.savefig('./sampling/random_hist.png')
plt.show()

########################################################################################################################
plt.style.use('default')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots()
ax.hist(bins[:-1], bins=bins,weights= counts_results['Uniform'],alpha=0.5, label='Uniform')
ax.hist(bins[:-1], bins=bins,weights= counts_results['Uniform Quality'],alpha=0.5, label='Uniform Quality')
ax.hist(bins[:-1], bins=bins,weights= counts_results['Uniform log Quality'],alpha=0.5, label='Uniform log Quality')
ax.set_xlabel('Net Present Value, $',fontname=typo)
ax.set_ylabel('Frequency',fontname=typo)
plt.legend()
plt.savefig('./sampling/uniform_hist.png')
plt.show()

########################################################################################################################
vals, names, xs = [],[],[]
for i, col in enumerate(samples_results.columns):
    vals.append(samples_results[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, samples_results[col].values.shape[0]))

plt.show()
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots()
ax.boxplot(vals, labels=names)
palette = ['r', 'g', 'b', 'y', 'k']
for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.05, color=c)
ax.set_xlabel('Sampling mthod',fontname=typo)
ax.set_ylabel('Net Present Value, $',fontname=typo)
plt.savefig('./sampling/boxplot.png')
plt.show()

########################################################################################################################
mag = int(args.num_of_train_sample / len(args.boundary_of_wells))
x1,y1,t1,l1 = np.array([]), np.array([]), np.array([]), np.array([])
for jj in range(len(args.boundary_of_wells)):
    xs, ys, ts,ls = np.array([]),np.array([]),np.array([]),np.array([])
    for s in samples_uniform_quality[jj * mag:(jj+1) * mag]:
        for w in s.wells:
            x, y, t, l = w.location['x'], w.location['y'], w.type['index'], w.type['label']
            xs = np.append(xs, np.array([x], dtype=np.int8))
            ys = np.append(ys, np.array([y], dtype=np.int8))
            ts = np.append(ts, np.array([t], dtype=np.int8))
            ls = np.append(ls, np.array([l]))

        x1 = np.append(x1, xs)
        y1 = np.append(y1, ys)
        t1 = np.append(t1, ts)
        l1 = np.append(l1, ls)

draw_wp(x1, y1, t1, f'_')
########################################################################################################################
def Summary_input(collection, threshold = None):
    '''
    Summary of trained model's results.
    :param collection: model collection of deep learning model object
    :param args: parsed arguments
    :param threshold: Set the threshold of R2 score
    :return: Sorted total R2 score list & list over threshold
    '''
    total_mdl = pd.DataFrame()
    for idx, i in enumerate(args.comb_dict.values()):
        total_mdl['_'.join(i)] = collection[idx].metric['r2_score']
    sorted_mdl = total_mdl.iloc[0,::].sort_values(ascending=False)
    sorted_mdl.name = 'R2'
    if not threshold:
        bmodel_sort = sorted_mdl
        threshold=0.9
        best_model_sort = sorted_mdl.T[sorted_mdl.T>threshold].dropna()
    else:
        best_model_sort = sorted_mdl.T[sorted_mdl.T>threshold].dropna()
        bmodel_sort = best_model_sort

    Dynamic = []
    Static = []
    OnlyDynamic = []
    OnlyStatic = []
    OnlyDynamicW = []
    OnlyStaticW = []
    Quality = []
    Hybrid = []
    Original = []
    WOHybrid = []
    for idx, i in enumerate(bmodel_sort.index):
        Dynamic.append('TOFP' in i or 'TOFP' in i or 'Sat' in i or 'Pressure' in i or 'ResPressure' in i)
        Static.append('Perm' in i or 'LogPerm' in i)
        Quality.append('Jang' in i)
        Hybrid.append(Dynamic[-1] and Static[-1] or Quality[-1])
        WOHybrid.append(Dynamic[-1] and Static[-1] and not Quality[-1])
        Original.append(set(['TOFP','TOFP','Config']) == set(i.split('_')))
        if set(['TOFP','TOFI','Config']) == set(i.split('_')): ranking = idx
        OnlyDynamic.append(Dynamic[-1] and not Static[-1] and not Quality[-1] and not 'Config' in i)
        OnlyStatic.append(Static[-1] and not Dynamic[-1] and not Quality[-1] and not 'Config' in i)
        OnlyDynamicW.append(Dynamic[-1] and not Static[-1] and not Quality[-1])
        OnlyStaticW.append(Static[-1] and not Dynamic[-1] and not Quality[-1])

    print('=================================================')
    print('================= S u m m a r y =================')
    print('=================================================')
    print('----------------- Original input ----------------')

    print(f"Is there the original input combinations? => {'Yes' if Original.count(True) else 'No'}")
    print(f'  Ranking :{ranking} \n' if Original.count(True) else '\n')
    print('----------------- Type analysis -----------------')
    print(f'  Num of Dynamic                    : {Dynamic.count(True)}')
    print(f'  Num of Static                     : {Static.count(True)}')
    print(f'  Num of Quality                    : {Quality.count(True)}')
    print(f'  Num of Hybrid                     : {Hybrid.count(True)}')
    print(f'  Num of Hybrid WO Quality          : {WOHybrid.count(True)}')
    print(f'  Num of Only Dynamic               : {OnlyDynamic.count(True)}')
    print(f'  Num of Only Static                : {OnlyStatic.count(True)}')
    print(f'  Num of Only Dynamic With Config   : {OnlyDynamicW.count(True)}')
    print(f'  Num of Only Static With Config    : {OnlyStaticW.count(True)}')
    print('\n')

    print('-------------- Individual analysis --------------')
    for input_ in args.input_list:
        print(f"  Num of {input_:13}: {[input_ in i for i in [idx.split('_') for idx in bmodel_sort.index]].count(True)}")
    print('\n')
    print('=================================================')

    return sorted_mdl, best_model_sort

########################################################################################################################
model_name = 'CNN'
args.input_flag = ('Config', 'TOFI', 'TOFP')
args.tensorboard_filename = '230126_JYKIM'
args.tensorboard_tagname = 'Loss (Original)'

# args.silent : iter_bar를 쓰고 싶을 땐 False, 아니면 True로 사용
args.silent = False
args.generator=None
Model_p = ProxyModel(args, samples_p, model_name=model_name)
if os.path.exists(f'{Model_p.saved_dir}/saved_model.pth'):
    # Loading only weight of trained model
    Model_p.model.load_state_dict(torch.load(f'{Model_p.saved_dir}/saved_model.pth'))
else:
    # Model_p.model = Model_p.train_model(samples_p, train_ratio=args.train_ratio,
    #                                     validate_ratio=args.validate_ratio,
    #                                     saved_dir=Model_p.saved_dir)
    # Saving trained proxy model as saved_model.pth (PyTorch Weight Information File)
    Model_p.model = Model_p.train_model(samples_p, train_ratio=args.train_ratio,
                                validate_ratio=args.validate_ratio,
                                saved_dir=Model_p.saved_dir,
                                saved_model='saved_model')

########################################################################################################################
# common things
args.num_of_generations = 100
args.num_of_particles = 40
args.tensorboard_tagname = f'Fitness(NPV)'

PSO_list = []
GWO_list = []
for run in range(4,11 ):
    args.tensorboard_filename = f'ECL/Run{run}'
    placement_positions = PlacementSample.make_candidate_solutions(num_of_candidates=args.num_of_particles)
    # GWO
    GWO_opt = GlobalOpt(args, placement_positions, [PlacementSample.perm[idx[0]-1] for idx in PlacementSample.perm_idx][2],
                    alg_name='GWO' , nn_model=None, sample=samples_p)
    GWO_opt.iterate(args.num_of_generations)

    # PSO
    PSO_opt = GlobalOpt(args, placement_positions, [PlacementSample.perm[idx[0]-1] for idx in PlacementSample.perm_idx][2],
                    alg_name='PSO' , nn_model=None, sample=samples_p)
    PSO_opt.iterate(args.num_of_generations)

    PSO_list.append(PSO_opt)
    GWO_list.append(GWO_opt)

########################################################################################################################
TLBO_list = []
TLBO_list_20 = []
for run in range(1,6):
    args.num_of_particles = 40
    args.num_of_generations = 50
    args.tensorboard_filename = f'ECL/Run{run}(popsize=40)'
    placement_positions = PlacementSample.make_candidate_solutions(num_of_candidates=args.num_of_particles)
    # TLBO with 40 populations
    TLBO_opt = GlobalOpt(args, placement_positions, [PlacementSample.perm[idx[0]-1] for idx in PlacementSample.perm_idx][2],
                    alg_name='TLBO', nn_model=None, sample=samples_p)
    TLBO_opt.iterate(args.num_of_generations)
    TLBO_list.append(TLBO_opt)

    args.num_of_particles = 20
    args.num_of_generations = 100
    args.tensorboard_filename = f'ECL/Run{run}(popsize=20)'
    placement_positions = PlacementSample.make_candidate_solutions(num_of_candidates=args.num_of_particles)
    # TLBO with 20 populations
    TLBO_opt_20 = GlobalOpt(args, placement_positions, [PlacementSample.perm[idx[0]-1] for idx in PlacementSample.perm_idx][2],
                    alg_name='TLBO', nn_model=None, sample=samples_p)
    TLBO_opt.iterate(args.num_of_generations)
    TLBO_list_20.append(TLBO_opt_20)
########################################################################################################################
DrawNPV(TLBO_list_20[-1], args)
args.num_of_particles = 40
args.num_of_generations = 5
args.tensorboard_filename = f'ECLF/Run{99}'
placement_positions = PlacementSample.make_candidate_solutions(num_of_candidates=args.num_of_particles)
# TLBO with 40 populations
TLBO_opt = GlobalOpt(args, placement_positions, [PlacementSample.perm[idx[0]-1] for idx in PlacementSample.perm_idx][2],
                alg_name='PSO', nn_model=None, sample=samples_p)
TLBO_opt.iterate(args.num_of_generations)
########################################################################################################################
seed(26727)
TLBO_list
for opt in TLBO_list_20:
    opt.algorithm.pbest = opt.algorithm.positions_all
    ViewParticles(opt, args,n_components=3,Npv_map=True,lag=20,filename=f'TLBO_3D_20.png')
    ViewParticles(opt, args,n_components=2,Npv_map=False,lag=20,filename=f'TLBO_2D_20.png')
    ViewParticles(opt, args,n_components=2,Npv_map=True,lag=20,filename=f'TLBO_2D_NPV.png')

seed(26727)
args.num_of_generations = 35
for i, opt in enumerate(TLBO_opt):
    opt.algorithm.pbest = opt.algorithm.positions_all
    ViewParticles(opt, args,n_components=2,Npv_map=False,lag=20,filename=f'GWO_2d_{i}.png')
########################################################################################################################
