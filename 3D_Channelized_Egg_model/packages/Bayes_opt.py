from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from packages.proxymodel import ProxyModel
from copy import copy
from packages.utils import fix_seed

class BayesOpt:
    def __init__(self,
                 args,
                 setting,
                 samples,
                 train_ratio=0.7,
                 validate_ratio=0.15,
                 seed=0
                 ):
        self.args = args
        self.setting = copy(setting)
        self.samples = samples
        self.train_ratio = train_ratio
        self.validate_ratio = validate_ratio
        self.hyperparameter = self.setting['Bayesian Optimization']
        #self.param_type = {'Batch_size': round, 'Epoch': round, 'Lr': float, 'Gamma': float}
        self.param_type = {'Batch_size': round, 'Epoch': round, 'Lr': float}
        self.setting['Silent'] = True
        self.seed = seed
    def _objective(self, Batch_size, Epoch, Lr):
        proxy_setting = self.setting
        Batch_size = int(pow(2,Batch_size))
        for param in self.hyperparameter.keys():
            proxy_setting[param] = self.param_type[param](eval(param))
        fix_seed(self.seed)
        model = ProxyModel(self.args, self.samples, setting=proxy_setting)
        model.model = model.train_model(self.samples, train_ratio=self.train_ratio,
                                        validate_ratio=self.validate_ratio)
        return model.metric['r2_score'][-1]

    def perform_BayesOpt(self, init_points=5, n_iter=20, acquisition="ei", param=0.0001):
        BO = BayesianOptimization(f= self._objective, pbounds=self.hyperparameter)
        acquisition_function = UtilityFunction(kind=acquisition, xi=param)
        BO.maximize(init_points=init_points, n_iter=n_iter, acquisition_function=acquisition_function)
        return BO

    def return_Param(self, opt_params, proxy_params):
        for param in self.hyperparameter.keys():
            proxy_params[param] = self.param_type[param](opt_params['params'][param])
        proxy_params['Batch_size'] = pow(2, proxy_params['Batch_size'])