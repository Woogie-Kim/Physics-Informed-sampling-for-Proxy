from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from packages.proxymodel import ProxyModel
from copy import copy
from packages.utils import fix_seed
import pickle
class BayesOpt:
    def __init__(self,
                 args,
                 setting,
                 samples,
                 train_ratio=0.7,
                 validate_ratio=0.15,
                 seed = 0
                 ):
        
        fix_seed(seed)
        self.args = args
        self.setting = copy(setting)
        self.samples = samples
        self.train_ratio = train_ratio
        self.validate_ratio = validate_ratio
        self.hyperparameter = self.setting['Bayesian Optimization']
        self.param_type = {'Batch_size': round, 'Epoch': round, 'Lr': float, 'Gamma': float}
        self.setting['Silent'] = True
        self.best_perf = 0
    def _objective(self, Batch_size, Epoch, Lr, Gamma):
        proxy_setting = self.setting
        for param in self.hyperparameter.keys():
            proxy_setting[param] = self.param_type[param](eval(param))
        model = ProxyModel(self.args, self.samples, setting=proxy_setting)
        model.model = model.train_model(self.samples, train_ratio=self.train_ratio,
                                        validate_ratio=self.validate_ratio)
        if model.metric['r2_score'][-1] >= self.best_perf:
            self.best_perf = model.metric['r2_score'][-1]
            with open(f'./cached/Proxy_best_Bayesopt.pkl', 'wb') as f:
                pickle.dump(model, f)

        return model.metric['r2_score'][-1]

    def perform_BayesOpt(self, init_points=5, n_iter=20, acquisition="ei", param=0.0001):
        BO = BayesianOptimization(f= self._objective, pbounds=self.hyperparameter)
        acquisition_function = UtilityFunction(kind=acquisition, xi=param)
        BO.maximize(init_points=init_points, n_iter=n_iter, acquisition_function=acquisition_function)
        return BO

    def return_Param(self, opt_params, proxy_params):
        for param in self.hyperparameter.keys():
            proxy_params[param] = self.param_type[param](opt_params['params'][param])