import os
import numpy as np
from tqdm.auto import tqdm
from packages.position import PositionExample
from packages.utils import *
from copy  import copy
from multiprocessing import Process
from time import sleep
from os.path import join
from os import remove, listdir
import matplotlib.pyplot as plt
import pickle


class DataExample:
    def __init__(self, position, fitness, violation, matrix, time, tof, pressure):
        self.position = position
        self.fitness = fitness
        self.violation = violation
        self.matrix = matrix
        self.time = time
        self.tof = tof
        self.pressure = pressure
        self.positions = None


class DataSampling:
    def __init__(self,
                 args,
                 ratio_of_infeasible=0.3,
                 wset=None,
                 location_fix=False,
                 type_fix=False,
                 drilling_time_fix=False,
                 control_fix=False,
                 well_type=None,
                 violation=None,
                 violation_check=True,
                 num_of_ensemble=None,
                 positions=None,
                 sampling_setting=None,
                 duplicate=None
                 ):
        self.args = args
        self.perm = load_matfile(args.perm_mat)
        self.perm_idx = load_matfile(args.perm_ID_mat, name="ens_selected")
        self.active = load_matfile(args.active_mat, name="ACTIVE")
        #self.perm = load_matfile(args.perm_mat, 'original')
        #self.perm_idx = load_matfile(args.perm_mat, 'selected')
        self.ratio_of_infeasible = ratio_of_infeasible
        if not wset:
            wset = [250, 350, 550, 650]     # default - Prod: 250~350 psi, Inj: 550~650 psi
        self.wset = wset
        self.type = type
        if not well_type:
            well_type = [-1, 0, 1]  # default - consider all type (-1: Inj. 0: No well, 1: Prod.)

        if duplicate:
            self.dup  = duplicate
        else:
            self.dup = [0]
        self.well_type = well_type
        self.location_fix = location_fix
        self.type_fix = type_fix
        self.drilling_time_fix = drilling_time_fix
        self.control_fix = control_fix
        self.violation = violation
        self.violation_check = violation_check
        self.num_of_ensemble = num_of_ensemble
        self.sampling_setting = sampling_setting
        self.available_method = args.available_method
        self.positions = positions

    def make_train_data(self, positions, perm, use_eclipse=True, use_frontsim=True, parallel=True, clean=True):
        max_process = self.sampling_setting['Process']
        # if not 'parallel' in locals():
        parallel = self.sampling_setting['Parallel']
        if parallel:
            if use_eclipse:
                ps = []
                for idx, position in enumerate(tqdm(positions, desc=f'now ecl simulate: ')):
                    p = Process(target=position.eclipse_parallel, args=(idx + 1, position, perm))
                    ps.append(p)
                    if len(ps) == max_process or idx + 1 == len(positions):
                        for p in ps: p.start()
                        for p in ps: p.join()
                        ps = []
                sleep(1)
                for idx in range(len(positions)): positions[idx].ecl_result(idx + 1, positions[idx])

            if use_frontsim:
                ps = []
                for idx, position in enumerate(tqdm(positions, desc=f'now frs simulate: ')):
                    p = Process(target=position.frontsim_parallel, args=(idx + 1, position, perm))
                    ps.append(p)
                    if len(ps) == max_process or idx + 1 == len(positions):
                        for p in ps: p.start()
                        for p in ps: p.join()
                        ps = []
                sleep(1)
                converted_list = []
                while not len(converted_list) >= len(positions):
                    for idx in range(len(positions)): positions[idx].converter(idx+1)
                    sleep(1)
                    for file in listdir(self.args.simulation_directory):
                        if 'F00' in file: converted_list.append(file)
                for idx in range(len(positions)): positions[idx].frs_result(idx + 1)

        else:
            for idx, position in enumerate(tqdm(positions, desc=f'now simulate: ')):
                if use_eclipse:
                    position.eclipse(idx+1, position, perm, mark=False)
                if use_frontsim:
                    position.frontsim(idx+1, position, perm)

        if clean:
            for file in listdir(position.simulation_directory):
                # if file.endswith(".RSM") or file.endswith(".F0001") or file.endswith(".F001") or file.endswith(".X0001")\
                #         or file.endswith(".X001"):
                if (not 'bat' in file) and (not 'txt' in file) and (not 'DATA' in file) and (not 'F00' in file) and (not 'RSM' in file):
                    remove(join(position.simulation_directory, file))

        return positions

    def make_candidate_solutions(self, num_of_candidates, location=None, type_real=None,
                                 drilling_time=None, control=None):
        """
        :param num_of_candidates: number of candidate solutions
        :param ratio_of_infeasible: infeasible means not satisfying defined constraints
        :param well_type: 1: "production", -1: "injection", 0: "no well".
                for consider all well type, well_type = [-1,0,1]
        :param type_fix: if type_fix True, then well_type must be defined
        :return: randomly initialized candidate solutions
        """
        # 2023-04-19 Masking
        mask_idx = np.random.choice(list(range(num_of_candidates)), int(num_of_candidates * self.ratio_of_infeasible), replace=False)
        mask = np.ones(num_of_candidates)
        mask[mask_idx] = 0
        positions = []
        if isinstance(location, list) and len(location) > 1:
            for idx in tqdm(range(self.sampling_setting['Number of Samples']), desc='Random Sampling...'):
                P = PositionExample(self.args, wset=self.wset, well_type=self.well_type, type_fix=self.type_fix,
                                    location_fix=self.location_fix, drilling_time_fix=self.drilling_time_fix,
                                    control_fix=self.control_fix, violation_check=self.violation_check,
                                    num_of_wells=self.sampling_setting['Number of max wells'],
                                    sampling_setting=self.sampling_setting)
                P.perm = self.sampling_setting['Permeability']
                P.initialize(ratio_of_infeasible=mask[idx], location=location[idx], type_real=type_real[idx],
                             drilling_time=drilling_time, control=control, duplicate=self.dup)
                self.dup.append([sorted(P.loc), sorted(P.t)])
                positions.append(P)
        else:
            if self.sampling_setting['Method'] in self.available_method:
                well_array = self.sampling_setting['Well array']
                sample_array = self.sampling_setting['Sample array']
                name = self.sampling_setting['Method']
                for i, n in enumerate(tqdm(well_array, desc=f"{name} sampling...")):
                    for idx in range(int(sample_array[i])):
                        P = PositionExample(self.args, wset=self.wset, well_type=self.well_type, type_fix=self.type_fix,
                                            location_fix=self.location_fix, drilling_time_fix=self.drilling_time_fix,
                                            control_fix=self.control_fix, violation_check=self.violation_check,
                                            num_of_wells=n, sampling_setting=self.sampling_setting)
                        P.perm = self.sampling_setting['Permeability']
                        P.initialize(ratio_of_infeasible=mask[idx], location=location, type_real=type_real,
                                     drilling_time=drilling_time, control=control, duplicate=self.dup)
                        self.dup.append([sorted(P.loc), sorted(P.t)])
                        positions.append(P)

            else:
                for idx in tqdm(range(self.sampling_setting['Number of Samples']), desc='Random Sampling...'):
                    P = PositionExample(self.args, wset=self.wset, well_type=self.well_type, type_fix=self.type_fix,
                                        location_fix=self.location_fix, drilling_time_fix=self.drilling_time_fix,
                                        control_fix=self.control_fix, violation_check=self.violation_check,
                                        num_of_wells=self.sampling_setting['Number of max wells'],
                                        sampling_setting=self.sampling_setting)
                    P.perm = self.sampling_setting['Permeability']
                    P.initialize(ratio_of_infeasible=mask[idx], location=location, type_real=type_real,
                                 drilling_time=drilling_time, control=control, duplicate=self.dup)
                    self.dup.append([sorted(P.loc), sorted(P.t)])
                    positions.append(P)
        return positions


def Packer(sampling_setting, args, permeability, load=True, save=True, train_data=True,use_frontsim=True, view=False, duplicate=None):
    if not duplicate:
        duplicate = [0]
    samples_p = []
    # 2023-01-03
    # If DataSample already exists, Just load DataSample. (\cached\sample_wp(5).pkl)
    if load:
        if os.path.exists(os.path.join(args.cached_dir, sampling_setting['Save'])):
            with open(os.path.join(args.cached_dir, sampling_setting['Save']), 'rb') as f:
                samples_p = pickle.load(f)
    else:
        if sampling_setting['Method'] == '2stage':
            # first stage
            setting_first = copy(sampling_setting)

            setting_first['Number of Samples'] = int(sampling_setting['Number of Samples'] / 2)
            setting_first['Sample array'] = setter(setting_first)
            # setting_first['Default radius'] = False
            PlacementSample_first = DataSampling(args, ratio_of_infeasible=sampling_setting['Infeasible ratio'],
                                                 wset=args.well_placement_wset, well_type=args.well_type,
                                                 location_fix=False, type_fix=False, drilling_time_fix=True,
                                                 control_fix=True, num_of_ensemble=args.num_of_ensemble,
                                                 sampling_setting=setting_first, duplicate=duplicate)

            first_p = PlacementSample_first.make_candidate_solutions(num_of_candidates=setting_first['Number of Samples'])
            if train_data:
                samples_p = PlacementSample_first.make_train_data(first_p, permeability, parallel=sampling_setting['Parallel'],
                                                                  use_frontsim=use_frontsim)
            else:
                samples_p.append(first_p)

            # first stage
            setting_second = copy(sampling_setting)
            setting_second['Number of Samples'] = int(sampling_setting['Number of Samples'] - setting_first['Number of Samples'] )
            pdf_npv, q_npv, nwell = get_CDF(samples_p, 0.75)
            setting_second['Quality'] = pdf_npv
            # setting_first['Default radius'] = True
            p_nwell = nwell/sum(nwell)
            lst = np.zeros_like(setting_second['Sample array'])
            for i in np.random.choice(list(sampling_setting['Well array']), setting_second['Number of Samples'], p=p_nwell):
                lst[i - sampling_setting['Well array'][0]] += 1
            setting_second['Sample array'] = lst
            PlacementSample_second = DataSampling(args, ratio_of_infeasible=sampling_setting['Infeasible ratio'],
                                                  wset=args.well_placement_wset, well_type=args.well_type,
                                                 location_fix=False, type_fix=False, drilling_time_fix=True,
                                                 control_fix=True, num_of_ensemble=args.num_of_ensemble,
                                                 sampling_setting=setting_second, duplicate=PlacementSample_first.dup)
            second_p = PlacementSample_second.make_candidate_solutions(num_of_candidates=setting_second['Number of Samples'])
            if train_data:
                samples_p += PlacementSample_second.make_train_data(second_p, permeability, parallel=sampling_setting['Parallel'],
                                                                    use_frontsim=use_frontsim)
            else:
                samples_p.append(second_p)

            if save:
                with open(os.path.join(args.cached_dir, sampling_setting['Save']), 'wb') as f:
                    pickle.dump(samples_p, f)

        else:
            PlacementSample = DataSampling(args, ratio_of_infeasible=sampling_setting['Infeasible ratio'],
                                           wset=args.well_placement_wset, well_type=args.well_type,
                                           location_fix=False, type_fix=False, drilling_time_fix=True,
                                           control_fix=True, num_of_ensemble=args.num_of_ensemble,
                                           sampling_setting=sampling_setting, duplicate=duplicate)
            initial_p = PlacementSample.make_candidate_solutions(num_of_candidates=sampling_setting['Number of Samples'])
            if train_data:
                samples_p = PlacementSample.make_train_data(initial_p, permeability, parallel=sampling_setting['Parallel'],
                                                            use_frontsim=use_frontsim)
            else:
                samples_p = initial_p
            if save:
                with open(os.path.join(args.cached_dir, sampling_setting['Save']), 'wb') as f:
                    pickle.dump(samples_p, f)
    if view:
        plt.hist([s.fit for s in samples_p])
        plt.show()
        preprocess_tof(samples_p)
    return samples_p

def Packer_new(sampling_setting, args, permeability, load=True, save=True, train_data=True, use_frontsim=True, view=False, duplicate=None):
    tmp_setting = copy(sampling_setting)
    if not duplicate:
        duplicate_short = [0]
        samples_p = []
    else:
        duplicate_short = [[sorted(s.loc), sorted(s.t)] for s in duplicate]
        samples_p = copy(duplicate)

    if load:
        if os.path.exists(os.path.join(args.cached_dir, sampling_setting['Save'])):
            with open(os.path.join(args.cached_dir, sampling_setting['Save']), 'rb') as f:
                samples_p = pickle.load(f)
    else:
        if (sampling_setting['Method'] == '2stage') or (sampling_setting['Method'] == 'physics'):
            tmp_setting['Use Quality'] = True

            if sampling_setting['Method'] == '2stage':
                pdf_npv, q_npv, nwell = get_CDF(duplicate, 0.75)
                tmp_setting['Quality'] = pdf_npv
                p_nwell = nwell/sum(nwell)
                lst = np.zeros(args.num_of_max_well - args.num_of_min_well + 1)
                for i in np.random.choice(list(tmp_setting['Well array']), tmp_setting['Number of Samples'], p=p_nwell):
                    lst[i - args.num_of_min_well] += 1
                tmp_setting['Sample array'] = lst

            elif sampling_setting['Method'] == 'physics':
                q_sl, p_well, potential = SLquality(args.permeability, duplicate, tmp_setting)
                nwell_sample = list(
                    np.random.choice(tmp_setting['Well array'], size=tmp_setting['Number of Samples'],
                                     p=p_well))
                tmp_setting['Quality'] = {}
                q_inj = quality(np.ones_like(quality(q_sl['I'], islog=False, smooth=False, view=False)))
                q_prod = quality(q_sl['P'], islog=False, smooth=False, view=False)
                if sum(q_inj) != 0: tmp_setting['Quality']['I'] = quality(np.ones_like(q_inj))
                else: tmp_setting['Quality']['I'] = q_inj
                if sum(q_prod) != 0: tmp_setting['Quality']['P'] = quality(np.ones_like(q_prod))
                else: tmp_setting['Quality']['P'] = q_inj
                tmp_setting['Sample array'] = [nwell_sample.count(j) for j in tmp_setting['Well array']]

            PlacementSample_second = DataSampling(args, ratio_of_infeasible=tmp_setting['Infeasible ratio'],
                                                  wset=args.well_placement_wset, well_type=args.well_type,
                                                 location_fix=False, type_fix=False, drilling_time_fix=True,
                                                 control_fix=True, num_of_ensemble=args.num_of_ensemble,
                                                 sampling_setting=tmp_setting, duplicate=duplicate_short)
            second_p = PlacementSample_second.make_candidate_solutions(num_of_candidates=tmp_setting['Number of Samples'])


            if train_data:
                samples_p += PlacementSample_second.make_train_data(second_p, permeability, parallel=sampling_setting['Parallel'],
                                                                    use_frontsim=use_frontsim)
            else:
                samples_p.append(second_p)

            if save:
                with open(os.path.join(args.cached_dir, sampling_setting['Save']), 'wb') as f:
                    pickle.dump(samples_p, f)

        else:
            if tmp_setting['Method'] == 'uniform':
                tmp_setting['Default radius'] = False
            else:
                tmp_setting['Default radius'] = True
            PlacementSample = DataSampling(args, ratio_of_infeasible=sampling_setting['Infeasible ratio'],
                                           wset=args.well_placement_wset, well_type=args.well_type,
                                           location_fix=False, type_fix=False, drilling_time_fix=True,
                                           control_fix=True, num_of_ensemble=args.num_of_ensemble,
                                           sampling_setting=sampling_setting, duplicate=duplicate_short)
            initial_p = PlacementSample.make_candidate_solutions(num_of_candidates=sampling_setting['Number of Samples'])
            if train_data:
                samples_p = PlacementSample.make_train_data(initial_p, permeability, parallel=sampling_setting['Parallel'],
                                                            use_frontsim=use_frontsim)
            else:
                samples_p = initial_p
            if save:
                with open(os.path.join(args.cached_dir, sampling_setting['Save']), 'wb') as f:
                    pickle.dump(samples_p, f)
    if view:
        plt.hist([s.fit for s in samples_p])
        plt.show()
    return samples_p