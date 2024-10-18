import copy
# LSTM 사용 시 WODataset도 필요함
# from dlmodels import WPDataset, WODataset
from packages.dlmodels import WPDataset, WODataset
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm.auto import tqdm
'''
Requirements for parallel simulation
'''
from packages.utils import preprocess_tof
from os.path import join
from os import remove, listdir
from multiprocessing import Process
from time import sleep
from sklearn.preprocessing import MinMaxScaler

########################################################################################################################
# Add a parents module for generalized meta heuristic optimizations
# Writer : Jongwook Kim
# Last update : 06-Feb-2023
#########################################################################################################################
class Meta_heuristic:
    def __init__(self,
                 args,
                 perms,
                 setting):
        self.args = args
        if len(perms) == args.num_of_x * args.num_of_y:
            self.perms = [perms]
        else:
            self.perms = perms
        self.setting = setting
        self.fit_true = []
    def evaluate(self, positions, neural_model=None):
        """
        :param positions:
        :param neural_model: pre-trained model
        :return:
        """
        args = self.args
        predictions_ens = []
        if neural_model:
            for k in self.perms:

                # if neural_model.model_name in ['CNN', 'ResNet']:
                #     if self.setting['Parallel']:
                #         ps = []
                #         for idx, position in enumerate(positions):
                #             p = Process(target=position.frontsim_parallel, args=(idx + 1, position, k))
                #             ps.append(p)
                #             if len(ps) == self.setting['Process'] or idx + 1 == len(positions):
                #                 for p in ps:
                #                     p.start()
                #                 for p in ps:
                #                     p.join()
                #                 ps = []
                if neural_model.model_name in ['CNN', 'ResNet']:
                    if self.setting['Parallel']:
                        ps = []
                        for idx, position in enumerate(positions):
                            p = Process(target=position.frontsim_parallel, args=(idx + 1, position, k))
                            ps.append(p)
                            p.start()
                            if len(ps) == self.setting['Process'] or idx + 1 == len(positions):
                                for p in ps:
                                    p.join()
                                ps = []
                            sleep(1)
                            converted_list = []
                            while not len(converted_list) >= len(positions):
                                for idx in range(len(positions)): positions[idx].converter(idx + 1)
                                sleep(1)
                                for file in listdir(self.args.simulation_directory):
                                    if file.endswith('.F0001'): converted_list.append(file)

                            for idx in range(len(positions)): positions[idx].frs_result(idx + 1)

                    else:
                        for idx, position in enumerate(positions):
                            position.frontsim(idx + 1, position, k)
                    for file in listdir(positions[0].simulation_directory):
                        remove(join(positions[0].simulation_directory, file))
                    # for file in listdir(position.simulation_directory):
                    #     if file.endswith(".RSM") or file.endswith(".F0001") or file.endswith(".F001") or file.endswith(
                    #             ".X0001") \
                    #             or file.endswith(".X001"):
                    #         remove(join(position.simulation_directory, file))
                    # cleaning
                    preprocess_tof(positions)
                    dataset = WPDataset(data=positions, maxtof=args.max_tof, maxP=args.max_pressure,
                                        res_oilsat=args.res_oilsat, nx=args.num_of_x, ny=args.num_of_y, transform=None,
                                        flag_input=neural_model.input)
                    # dataset = WPDataset(positions, args.max_tof, args.num_of_x, args.num_of_y, None)
                elif neural_model.model_name == 'LSTM':

                    dataset = WODataset(positions, args.production_time, args.dstep, args.tstep, None)
                    # dataset = WPDataset(positions, args.production_time, args.dstep, args.tstep, None)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                predictions, _ = neural_model.inference(neural_model.model, dataloader, label_exist=False)
                predictions_ens.append([p for p in predictions])

        else:
            for k in self.perms:
                predictions = []
                if self.setting['Parallel']:
                    ps = []
                    for idx, position in enumerate(positions):
                        p = Process(target=position.eclipse_parallel, args=(idx + 1, position, k))
                        ps.append(p)
                        if len(ps) == self.setting['Process'] or idx + 1 == len(positions):
                            for p in ps: p.start()
                            for p in ps: p.join()
                            ps = []
                    sleep(1)
                    for idx in range(len(positions)): positions[idx].ecl_result(idx + 1, positions[idx])
                    for file in listdir(position.simulation_directory):
                        if file.endswith(".RSM"): remove(join(position.simulation_directory, file))
                    for position in positions:
                        predictions.append(position.fit)
                    predictions_ens.append(predictions)

                else:
                    for idx, position in enumerate(positions):
                        fit, _ = position.eclipse(idx + 1, position, k)
                        predictions.append(fit)
                    predictions_ens.append(predictions)

        predictions = np.mean(np.array(predictions_ens), axis=0).squeeze().tolist()
        for position, pred in zip(positions, predictions):
            position.fit = pred


        return positions


    def get_true(self, positions):
        predictions_ens = []
        for k in self.perms:
            predictions = []
            if self.setting['Parallel']:
                ps = []
                for idx, position in enumerate(positions):
                    p = Process(target=position.eclipse_parallel, args=(idx + 1, position, k))
                    ps.append(p)
                    if len(ps) == self.setting['Process'] or idx + 1 == len(positions):
                        for p in ps: p.start()
                        for p in ps: p.join()
                        ps = []
                sleep(1)
                for idx in range(len(positions)):
                    _, fit_ecl = positions[idx].ecl_result(idx + 1, positions[idx], mark=False)
                    predictions.append(fit_ecl)
                for file in listdir(position.simulation_directory):
                    if file.endswith(".RSM"): remove(join(position.simulation_directory, file))
                for position in positions:
                    predictions_ens.append(predictions)

            else:
                for idx, position in enumerate(positions):
                    fit_ecl, _ = position.eclipse(idx + 1, position, k)
                    predictions.append(fit_ecl)
                predictions_ens.append(predictions)

        predictions = np.mean(np.array(predictions_ens), axis=0).squeeze().tolist()
        return predictions

    def _find_best_position(self, positions):
        dominate = self.check_domination(positions)
        return positions[dominate[0]]

    def check_domination(self, positions):
        dominated_idx = []
        for idxA, positionA in enumerate(positions):
            for idxB, positionB in enumerate(positions):
                if idxA == idxB:
                    continue
                dominate = self._filter_method(positionA, positionB)
                if dominate == 1:
                    dominated_idx.append(idxB)
        dominated_idx = set(dominated_idx)
        all_idx = set(range(0, len(positions)))

        return list(all_idx.difference(dominated_idx))

    def _filter_method(self, A, B):
        if A.violation > 0 and B.violation > 0:
            dominate = int(A.violation < B.violation)
        elif any([v == 0 for v in [A.violation, B.violation]]) and not all(
                [v == 0 for v in [A.violation, B.violation]]):
            dominate = int(A.violation == 0)
            # dominate = int(A.violation == 0)
        else:  # consider multi-objectives
            if not isinstance(A.fit, list):
                A_fit, B_fit = [A.fit], [B.fit]
            else:
                A_fit, B_fit = A.fit, B.fit
            dominate = all([a >= b for a, b in zip(A_fit, B_fit)]) \
                       and any([a > b for a, b in zip(A_fit, B_fit)])
        return dominate

########################################################################################################################
# integrating codes
# Modifier : Jongwook Kim
# Last update : 06-Jan-2023
########################################################################################################################
class PSO(Meta_heuristic):
    def __init__(self,
                 args,
                 positions,
                 perms,
                 setting,
                 index=False):
        super(PSO, self).__init__(args, perms, setting)
        self.args = args
        self.setting = setting
        self.w = args.w
        self.c1 = args.c1
        self.c2 = args.c2
        self.index = index
        self.positions_all = [positions]
        self.positions_particle = [[position] for position in positions]
        self.gbest = []
        self.pbest = []
        if len(perms) == args.num_of_x * args.num_of_y:
            self.perms = [perms]
        else:
            self.perms = perms

    def update(self, positions):
        positions_next = copy.deepcopy(positions)
        # update violation for filter method
        for position_new in positions_next:
            position_new.violation = position_new.violate(position_new.wells, True)

        #positions_particle = copy.deepcopy(self.positions_particle)
        positions_particle = self.positions_particle
        for particle_idx in range(self.setting['Number of particles']):
            positions_particle[particle_idx].append(positions_next[particle_idx])

        # find a personal best position
        pbest_position = [self._find_best_position(positions_particle[particle_idx]) for particle_idx in range(self.setting['Number of particles'])]
        self.pbest.append(pbest_position)

        # update a global best position
        gbest_position = self._get_gbest_position()
        self.gbest.append(gbest_position)

        positions_next_ = copy.deepcopy(positions_next)
        for particle_idx, position in enumerate(positions_next_):
            position = self._cal_velocity_and_update(position, particle_idx)
            for w in position.wells:
                w._cut_boundaries()
            # update violation
            position.violation = position.violate(position.wells, True)

        self.positions_all.append(positions_next_)
        for particle_idx in range(self.setting['Number of particles']):
            self.positions_particle[particle_idx].append(positions_next_[particle_idx])

    def _cal_velocity_and_update(self, position, particle_idx):
        pbest_position = self.pbest[-1]
        gbest_position = self.gbest[-1]

        attributes = position.wells[0].attributes
        if self.index == True: attributes['location'] = ['index']
        # w, c1, c2 = self.w, self.c1 * random.random(), self.c2 * random.random()
        for well, well_p, well_g in zip(position.wells, pbest_position[particle_idx].wells, gbest_position.wells):
            for var, elem in attributes.items(): # attr = location, var = ['x', 'y', 'z']
                current = getattr(well, var)
                pbest = getattr(well_p, var)
                gbest = getattr(well_g, var)
                if not current:     # not defined variable type
                    continue
                for e in elem:
                    if current[e] and not e == 'z':
                        w, c1, c2 = self.w, self.c1 * random.random(), self.c2 * random.random()
                        vel = w * current['velocity'][e] + c1*(pbest[e]-current[e]) + c2*(gbest[e]-current[e])
                        current['velocity'][e] = vel
                        current[e] += current['velocity'][e]
                if self.index == True:
                    if well.location['index'] >= 3600: well.location['index'] = 3600
                    elif well.location['index'] <= 1: well.location['index'] = 1
                    else: well.location['index'] = int(well.location['index'])
                    well._index_to_coord(well.location['index'])


        return position

    def _get_gbest_position(self):
        if not self.gbest:
            return self._find_best_position(self.pbest[-1])
        elif self.check_domination([self.gbest[-1], self._find_best_position(self.pbest[-1])])[0] == 0:
            return self.gbest[-1]
        else:
            return self._find_best_position(self.pbest[-1])

########################################################################################################################
# Add genetic algorithm
# Modifier : Jongwook Kim
# Last update : 06-Feb-2023
########################################################################################################################
class GA(Meta_heuristic):
    def __init__(self,
                 args,
                 positions,
                 perms,
                 setting):
        super(GA, self).__init__(args, perms, setting)
        self.args = args
        self.setting = setting
        self.crossover_rate = 0.9
        self.mutation_rate = 0.05
        self._make_chromosome(positions)
        self.positions_all = [positions]
        self.gbest = []
        self.num_of_particles = self.setting['Number of particles']
        self.num_of_parents = int(self.num_of_particles * self.crossover_rate)
        if not args.selection_type: self.selection_type = 'tournament'
        self.selection_type = args.selection_type
        if len(perms) == args.num_of_x * args.num_of_y: self.perms =[perms]
        else: self.perms = perms

    def update(self, positions):
        positions_next = copy.deepcopy(positions)

        # update violation for filter method
        for position_new in positions_next:
            position_new.violation = position_new.violate(position_new.wells, True)

        # update a global best position(elite)
        gbest = self._get_gbest_position()
        self.gbest.append(copy.deepcopy(gbest))

        # encoding
        self._make_chromosome(positions_next)

        # Step 2. parents selection
        parents = self._selection(positions_next)

        # Step 3. crossover
        # crossover child(0.9) + best parents(0.1)
        children = self._crossover(parents)
        children += self._parents_best(parents, self.num_of_particles - len(children))

        # Step 4. mutation
        mutated_children = self._mutation(children)

        # Step 5. decoding
        self._chromosome_to_position(mutated_children)

        # update violation
        for position in mutated_children:
            position.violation = position.violate(position.wells, True)

        self.positions_all.append(mutated_children)

    def _make_chromosome(self, samples):
        for pos in samples:
            chromosome = {'x': [], 'y': [], 'type': [], 'vector': []}
            for well in pos.wells:
                chromosome['x'].append(well.location['x'])
                chromosome['y'].append(well.location['y'])
                chromosome['type'].append(well.type['index'])
                chromosome['vector'] += [well.location['x'], well.location['y'], well.type['index']]
            pos.chromosome = chromosome

    def _selection(self, samples):
        # next_gen = deepcopy(self.gen[-1])
        if self.selection_type == 'roulette':
            pop, _ = self._Roulette_Wheel(samples)
        elif self.selection_type == 'tournament':
            pop = self._tournament(samples)
        parents = copy.deepcopy(list(np.random.choice(pop, self.num_of_parents, replace=False)))
        return parents

    def _crossover(self, parents):
        couples = np.random.choice(parents, [len(parents) // 2, 2], replace=False)
        children = []
        for couple in couples:
            c1, c2 = couple
            cv_point = 3 * np.random.randint(1,15)
            c1.chromosome['vector'][cv_point:], c2.chromosome['vector'][:cv_point] = c2.chromosome['vector'][cv_point:], \
                                                                                     c1.chromosome['vector'][:cv_point]
            children += c1, c2
        return children

    def _parents_best(self, parents, num_of_elite):
        pop = copy.deepcopy(parents)
        parents_best = []
        for _ in range(num_of_elite):
            current_best = self._find_best_position(pop)
            parents_best.append(current_best)
            pop.remove(current_best)
        return parents_best

    def _mutation(self, children):
        for child in children:
            for idx, parameter in enumerate(child.chromosome['vector']):
                if random.random() > self.crossover_rate:
                    if idx < 2 * self.args.num_of_max_well:
                        parameter = np.random.randint(1, self.args.num_of_x)
                    else:
                        parameter = np.random.choice(list(self.args.well_type.values()))
        return children

    def _chromosome_to_position(self, samples):
        for pos in samples:
            vec = np.array(pos.chromosome['vector']).reshape(-1, 3).T
            for idx, well in enumerate(pos.wells):
                well.location['x'] = pos.chromosome['x'][idx] = vec[0, idx]
                well.location['y'] = pos.chromosome['y'][idx] = vec[1, idx]
                well.type['index'] = pos.chromosome['type'][idx] = vec[-1, idx]
                well.type['label'] = pos._convert_type_value_to_str(well.type['index'])

                # Roulette wheel
    #
    # def _find_best_position(self, positions):
    #     dominate = check_domination(positions)
    #     return positions[dominate[0]]

    def _Roulette_Wheel(self, particles):
        fits = []
        scaler = MinMaxScaler()
        for particle in particles:
            fits.append(particle.fit)
        scaler.fit(np.array(fits).reshape(-1, 1))
        scaled_fits = scaler.transform(np.array(fits).reshape(-1, 1))
        # percentile of roulette
        p = scaled_fits.flatten() / sum(scaled_fits.flatten())
        return list(np.random.choice(particles, self.args.num_of_particles, p=p)), p

        # Tournament selection

    def _tournament(self, particles):
        tournament_table = []
        for idx, pop in enumerate(particles):
            p = np.ones_like(particles, dtype=int) / (len(particles) - 1)
            p[idx] = 0
            opponent = np.random.choice(particles, p=p)
            tournament_table.append(self._find_best_position([pop, opponent]))
        return tournament_table

    def _get_gbest_position(self):
        if not self.gbest:
            return self._find_best_position(self.positions_all[-1])
        elif self.check_domination([self.gbest[-1], self._find_best_position(self.positions_all[-1])])[0] == 0:
            return self.gbest[-1]
        else:
            return self._find_best_position(self.positions_all[-1])


########################################################################################################################
# Add grey wolf optimization
# Modifier : Jongwook Kim
# Last update : 08-Feb-2023
########################################################################################################################
class Wolf:
    def __init__(self):
        self.fit = -float("inf")
        self.violation = float("inf")

class GWO(Meta_heuristic):
    def __init__(self,
                 args,
                 positions,
                 perms,
                 setting):
        super(GWO, self).__init__(args, perms)
        self.args = args
        self.setting = setting
        self.positions_all = [positions]
        self.gbest = []
        self.alpha = [Wolf()]
        self.beta = [Wolf()]
        self.delta = [Wolf()]

        if len(perms) == args.num_of_x * args.num_of_y:
            self.perms = [perms]
        else:
            self.perms = perms


    def update(self, positions):
        positions_next = copy.deepcopy(positions)
        # update violation for filter method
        for position_new in positions_next:
            position_new.violation = position_new.violate(position_new.wells, True)

        # Update Alpha, Beta, and Delta
        self._set_hierarchy(positions_next)

        # Update gbest for generalization
        self.gbest.append(copy.deepcopy(self.alpha[-1]))

        # Update parameter a (a decreases linearly fron 2 to 0)
        a = 2 - len(self.positions_all) * (2 / self.args.num_of_generations)

        positions_next_ = copy.deepcopy(positions_next)
        for position in positions_next_:
            position = self._encircling_and_hunting(position, a)
            for w in position.wells:
                w._cut_boundaries()
            # update violation
            position.violation = position.violate(position.wells, True)

        self.positions_all.append(positions_next_)

    def _encircling_and_hunting(self, position, a):
        alpha_position = self.alpha[-1]
        beta_position = self.beta[-1]
        delta_position = self.delta[-1]

        attributes = position.wells[0].attributes

        for well, well_a, well_b, well_d in zip(position.wells, alpha_position.wells, beta_position.wells,
                                                delta_position.wells):
            for var, elem in attributes.items():  # attr = location, var = ['x', 'y', 'z']
                current = getattr(well, var)
                abest = getattr(well_a, var)
                bbest = getattr(well_b, var)
                dbest = getattr(well_d, var)
                if not current:  # not defined variable type
                    continue
                for e in elem:
                    if current[e] and not e == 'z':
                        r1, r2 = random.random(), random.random()
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_alpha = abs(C1 * abest[e] - current[e])
                        X1 = abest[e] - A1 * D_alpha

                        r1, r2 = random.random(), random.random()
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        D_beta = abs(C2 * bbest[e] - current[e])
                        X2 = bbest[e] - A2 * D_beta

                        r1, r2 = random.random(), random.random()
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        D_delta = abs(C3 * dbest[e] - current[e])
                        X3 = dbest[e] - A3 * D_delta

                        current[e] = (X1 + X2 + X3) / 3
        return position

    def _set_hierarchy(self, positions_next):
        for position in positions_next:
            if self.check_domination([position, self.alpha[-1]])[0] == 0:
                self.delta.append(copy.deepcopy(self.beta[-1]))
                self.beta.append(copy.deepcopy(self.alpha[-1]))
                self.alpha.append(copy.deepcopy(position))

            elif self.check_domination([position, self.beta[-1]])[0] == 0:
                self.delta.append(copy.deepcopy(self.beta[-1]))
                self.beta.append(copy.deepcopy(position))

            elif self.check_domination([position, self.delta[-1]])[0] == 0:
                self.delta.append(copy.deepcopy(position))


########################################################################################################################
# Add Teaching-learning based optimization
# Modifier : Jongwook Kim
# Last update : 12-Feb-2023
########################################################################################################################
class TLBO(Meta_heuristic):
    def __init__(self,
                 args,
                 positions,
                 perms,
                 index=False):
        super(TLBO, self).__init__(args, perms)
        self.args = args
        self.index = index
        self.positions_all = [positions]
        self.gbest = []
        if len(perms) == args.num_of_x * args.num_of_y:
            self.perms = [perms]
        else:
            self.perms = perms

    def update(self, positions):
        positions_initial = copy.deepcopy(positions)
        # update violation for filter method
        for position in positions_initial:
            position.violation = position.violate(position.wells, True)

        if len(self.positions_all) >= 2:
            positions_before = copy.deepcopy(self.positions_all[-2])
            for position in positions_before:
                position.violation = position.violate(position.wells, True)

            # Greedy selection
            positions_init = []
            for old, new in zip(positions_before, positions_initial):
                if self.check_domination([old, new])[0] == 0:
                    positions_init.append(old)
                else:
                    positions_init.append(new)
            positions_initial = positions_init
            self.positions_all[-1] = positions_init

        # teacher phase
        # find a teacher position
        teacher = self._get_teacher_position(positions_initial)
        self.gbest.append(teacher)

        # Learned by teacher
        positions_teaching = copy.deepcopy(positions_initial)
        teaching_score = self._get_mean(positions_teaching)

        for position in positions_teaching:
            position = self._teach(position, teaching_score)
            for w in position.wells:
                w._cut_boundaries()
            # update violation
            position.violation = position.violate(position.wells, True)

        # evaluate
        positions_teaching = self.evaluate(positions_teaching)

        # Greedy selection
        positions_teached = []
        for old, new in zip(positions_initial, positions_teaching):
            if self.check_domination([old, new])[0] == 0:
                positions_teached.append(copy.deepcopy(old))
            else:
                positions_teached.append(copy.deepcopy(new))

        # learner phase
        for idx, learner in enumerate(positions_teached):
            mate = np.random.choice(positions_teached[:idx] + positions_teached[idx + 1:])
            learner = self._study_with_mate(learner, mate)
            for w in learner.wells:
                w._cut_boundaries()
            learner.violation = learner.violate(learner.wells, True)

        self.positions_all.append(positions_teached)

    def _teach(self, position, learners):
        teacher_position = self.gbest[-1]

        # add learner's vector
        attributes = position.wells[0].attributes

        t_f = round(1 + random.random())
        r = random.random()
        for idx, (well, well_t) in enumerate(zip(position.wells, teacher_position.wells)):
            for var, elem in attributes.items():  # attr = location, var = ['x', 'y', 'z']
                current = getattr(well, var)
                teacher_ = getattr(well_t, var)
                if not current:  # not defined variable type
                    continue
                for e in elem:
                    if current[e] and not e == 'z':
                        mean_learners = np.mean(learners[idx][var][e])
                        diff = r * (teacher_[e] - t_f * mean_learners)
                        current[e] += diff
        return position

    def _get_mean(self, positions):
        # add learner's vector
        attributes = positions[0].wells[0].attributes
        learners = []
        for position in positions:
            for idx, well in enumerate(position.wells):
                learners.append({var: {e: [] for e in elem} for var, elem in attributes.items()})
                for var, elem in attributes.items():  # attr = location, var = ['x', 'y', 'z']
                    current = getattr(well, var)
                    if not current:  # not defined variable type
                        continue

                    for e in elem:
                        if current[e] and not e == 'z':
                            learners[idx][var][e].append(current[e])
        return learners

    def _study_with_mate(self, learner, mate):
        isdominate = self.check_domination([learner, mate])[0] == 0
        return self.__learn__([learner, mate], isdominate)[0]

    def __learn__(self, bin_positions, isdominate):
        attributes = bin_positions[0].wells[0].attributes
        r = random.random()
        for well_1, well_2 in zip(bin_positions[0].wells, bin_positions[1].wells):
            for var, elem in attributes.items():  # attr = location, var = ['x', 'y', 'z']
                learner_1 = getattr(well_1, var)
                learner_2 = getattr(well_2, var)
                if not learner_1:  # not defined variable type
                    continue
                for e in elem:
                    if learner_1[e] and not e == 'z':
                        diff = r * (learner_1[e] - learner_2[e])
                        if isdominate:
                            learner_1[e] += diff
                        else:
                            learner_1[e] -= diff
        return bin_positions


    def _get_teacher_position(self, positions):
        if not self.gbest:
            return self._find_best_position(positions)
        elif self.check_domination([self.gbest[-1], self._find_best_position(positions)])[0] == 0:
            return self.gbest[-1]
        else:
            return self._find_best_position(positions)

