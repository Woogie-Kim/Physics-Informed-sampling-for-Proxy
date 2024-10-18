from packages import algorithms
from tqdm.auto import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter

########################################################################################################################
# integrating codes
# Modifier : Jongwook Kim
# Last update : 01-Jan-2023
########################################################################################################################
# initialization changes
# add
# + self.fine_tune : determine whether retraining process is fine-tuning or not, boolean type of instance
########################################################################################################################
class GlobalOpt:
    def __init__(self,
                 args,
                 positions,
                 perms,
                 setting,
                 sample=None,
                 ):
        self.args = args
        self.sample = sample
        self.setting = setting
        self.alg_name = setting['Algorithm']
        self.nn_model = setting['Proxy']
        self.fine_tune = setting['Fine Tune']

        if len(perms) == args.num_of_x * args.num_of_y: self.perms = [perms]
        else: self.perms = perms
        self.algorithm = getattr(algorithms, self.alg_name)(args,
                                                       positions,
                                                       perms,
                                                       setting)
    def iterate(self, num_of_generations, eval=True):
        if self.setting['Tensorboard']['Use']:
            writer = SummaryWriter(f"./logs/Opt/{self.alg_name}/{self.setting['Tensorboard']['Filename']}")
        # consider initial
        iterbar = tqdm(range(num_of_generations + 1))
        for gen in iterbar:
            positions_evaluated = self.algorithm.evaluate(self.algorithm.positions_all[-1], self.nn_model)
            self.algorithm.update(positions_evaluated)
            if self.setting['Tensorboard']['Use']:
                writer.add_scalar(self.setting['Tensorboard']['Tagname'], self.algorithm.gbest[-1].fit, gen)
            iterbar.set_description(f"best fit = {self.algorithm.gbest[-1].fit/1e6:.2f} MM$")


            if self.nn_model:
                if self.setting['Use retrain']:
                    if gen in range(self.setting['Span of retrain'], num_of_generations + 1, self.setting['Span of retrain']):
                        print(f"now retraining")
                        self._retrain(self.algorithm.positions_all[-1])
        if self.setting['Tensorboard']['Use']:
            writer.close()
        if self.nn_model:
            if eval:
                # self.fits_true = {'gbest':[],'pbest':[]}
                # self.fits_proxy = {'gbest': [], 'pbest': []}
                self.fits_true = {'gbest': []}
                self.fits_proxy = {'gbest': []}
                print('Evaluate gbest...')
                fit_true = self.algorithm.get_true(self.algorithm.gbest)
                self.fits_true['gbest'] =fit_true
                self.fits_proxy['gbest'] = [gbest.fit for gbest in self.algorithm.gbest]
                # for idx, pbest in tqdm(enumerate(self.algorithm.pbest), desc='Evaluate pbest...'):
                #     fit_true = self.algorithm.get_true(pbest)
                #     self.fits_true['pbest'].append(fit_true)
                #     self.fits_proxy['pbest'].append([pbest.fit for pbest in self.algorithm.pbest[idx]])
        else:
            self.fits = [gbest.fit for gbest in self.algorithm.gbest]

    def get_solution(self, location=False, type=False, drilling_time=False, control=False):
        best = {}
        if location:
            best['location'] = [w.location['index'] for w in self.algorithm.gbest[-1].wells if w.type['index'] != 0]
        if type:
            best['type'] = [w.type['index'] for w in self.algorithm.gbest[-1].wells if w.type['index'] != 0]
        if drilling_time:
            best['drilling_time'] = [w.drilling_time['time']
                                     for w in self.algorithm.gbest[-1].wells if w.type['index'] != 0]
        if control:
            best['control'] = [(w.well_control['alpha'], w.well_control['beta'], w.well_control['gamma'])
                               for w in self.algorithm.gbest[-1].wells if w.type['index'] != 0]

        return best


########################################################################################################################
# _retrain changes
# add fine tuning
########################################################################################################################
    def _retrain(self, sample):
        # TODO XXX
        args = self.args
        if self.nn_model:
            # 잘못되어도 단단히 잘못됨
            # self.sample += sample * args.num_of_ensemble

            new_positions = copy.deepcopy(sample)
            new_sample = []
            for perm in self.perms:
                for idx, position in enumerate(new_positions):
                    position.eclipse(idx + 1, position, perm)
                new_sample += new_positions
            self.sample = self.sample + new_sample

            if self.fine_tune:
                for conv in self.nn_model.layer.parameters():
                    conv.requires_grad = False
                for fc in self.nn_model.fc_layer.parameters():
                    fc.requires_grad = True

            self.nn_model.model = self.nn_model.train_model(self.sample, train_ratio=args.train_ratio,
                                                            validate_ratio=args.validate_ratio,
                                                            saved_dir=self.nn_model.saved_dir)
