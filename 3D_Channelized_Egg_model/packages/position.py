from packages.utils import *
from packages.simulate import Simulate
import numpy as np
import random
from math import *
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.special import softmax
from itertools import product


class WellExample:
    def __init__(self, args,
                 location=None,
                 type=None,
                 drilling_time=None,
                 well_control=None,
                 well_types=None,
                 ):
        self.args = args

        self.attributes = {'location': ['x', 'y', 'z'], 'type': ['type'],
                           'drilling_time': ['time'], 'well_control': ['alpha', 'beta', 'gamma']}

        self.location = dict()
        self.type = dict()
        self.drilling_time = dict()
        self.well_control = dict()

        self.well_types = well_types

        if location:  # location is provided
            self._location(location)

        if type:
            self._type(type)

        if drilling_time:
            self._drilling_time(drilling_time)

        if well_control:
            self._well_control(well_control)

    def _index_to_coord(self, index):
        """
        :param index: well location index
        :return: without return. set (x,y,z) coordinates, boundaries
        """
        # 2023-05-09 indexing 들어가는 것 eclipse 방식으로 수정
        nx, ny, nz = self.args.num_of_x, self.args.num_of_y, self.args.num_of_z
        self.location['x'] = np.mod(index, nx) + ((np.mod(index, nx) == 0) * nx)
        self.location['y'] = ceil(index / nx)
        while self.location['y'] > ny:
            self.location['y'] -= ny
        self.location['z'] = ceil(index / nx / ny)
    def _location(self, index):
        """
        :param index: well location index
        :return: without return. set (x,y,z) coordinates, boundaries
        """
        # 2023-05-09 indexing 들어가는 것 eclipse 방식으로 수정
        nx, ny, nz = self.args.num_of_x, self.args.num_of_y, self.args.num_of_z
        self.location['x'] = np.mod(index, nx) + ((np.mod(index, nx) == 0) * nx)
        self.location['y'] = ceil(index / nx)
        while self.location['y'] > ny:
            self.location['y'] -= ny
        self.location['z'] = ceil(index / nx / ny)
        boundary = dict()
        for coord, num in zip(['x', 'y', 'z'], [nx, ny, nz]):
            boundary[coord] = [1, num]
        self.location['boundary'] = boundary
        self.location['index'] = index
        velocity = dict()
        for coord in self.attributes['location']:
            velocity[coord] = 0.0
        # 23-02-06 for optimization using index parameter
        velocity['index'] = 0.0
        self.location['velocity'] = velocity

    def _type(self, type):
        well_types = self.well_types

        def __set_type_boundaries__():
            boundary = dict()
            values = list(well_types.values())
            label = list(well_types.keys())
            s = values[0]
            idx = 0
            while True:
                if len(boundary) == len(values) - 1:
                    # 2023-04-28 const 관련 (only prod. well)
                    if len(values) == 1: boundary[label[idx]] = (s, s + 1e-4)
                    else: boundary[label[idx]] = (s, values[-1])
                    break
                e = s + (values[-1] - values[0]) / len(values)

                boundary[label[idx]] = (s, e)
                s = e
                idx += 1
            return boundary

        self.type['label'] = type
        self.type['index'] = self.well_types[type]
        self.type['boundary'] = __set_type_boundaries__()
        self.type['type'] = self.type['boundary'][type][0] + \
                            (self.type['boundary'][type][1] - self.type['boundary'][type][0]) * random.random()
        velocity = dict()
        for elem in self.attributes['type']:
            velocity[elem] = 0.0
        self.type['velocity'] = velocity

    def _drilling_time(self, drilling_time):
        self.drilling_time['time'] = drilling_time
        self.drilling_time['boundary'] = {'time': [0, 1]}
        velocity = dict()
        for elem in self.attributes['drilling_time']:
            velocity[elem] = 0.0
        self.drilling_time['velocity'] = velocity

    def _well_control(self, control):
        self.well_control['alpha'] = control[0]
        self.well_control['beta'] = control[1]
        self.well_control['gamma'] = control[2]
        boundary = dict()
        for coeff, num in zip(self.well_control.keys(), [1, pi, pi]):
            boundary[coeff] = [0, num]
        self.well_control['boundary'] = boundary
        velocity = dict()
        for elem in self.attributes['well_control']:
            velocity[elem] = 0.0
        self.well_control['velocity'] = velocity

    def _set_control(self, control_range, time_steps):
        low, high = control_range[self.type['label']]
        if self.well_control:   # by Awotunde (2014)
            alpha, beta, gamma = self.well_control['alpha'], self.well_control['beta'], self.well_control['gamma']
            time_steps = [t / (self.args.production_time/self.args.dstep) for t in time_steps]
            control = [(high-low)/2*alpha*(1+cos(beta*(1-t)+gamma))+low for t in time_steps]
        else:                   # relaxation
            if not self.type['label'] == 'No':
                boundary = self.type['boundary'][self.type['label']]
                if self.type['label'] == 'P':
                    control = high - (self.type['type'] - boundary[0]) / (boundary[1] - boundary[0]) * (high - low)
                elif self.type['label'] == 'I':
                    control = low + (-self.type['type'] - boundary[1]) / (boundary[1] - boundary[0]) * (high - low)
            else:
                control = sum(control_range['No'])/len(control_range['No'])
            control = [control] * len(time_steps)
        self.control = control

    def _set_schedule(self, time_steps):
        if self.well_control:
            drilling_time = ceil(self.args.production_time/self.args.dstep * self.drilling_time['time'])
            schedule = [0] * (drilling_time-1) + [1] * (len(time_steps) - (drilling_time-1))
        else:
            schedule = [1] * len(time_steps)
        self.schedule = schedule

    def _cut_boundaries(self):
        attributes = self.attributes

        for var, elem in attributes.items():
            func = getattr(self, var)
            if 'boundary' in list(func.keys()):
                for e in elem:
                    if not e == 'index':
                        if not var == 'type':
                            lower, upper = func['boundary'][e][0], func['boundary'][e][1]
                        else:
                            lower, upper = min(min(func['boundary'].values())) + 1e-12, \
                                           max(max(func['boundary'].values())) - 1e-12
                        if func[e]:
                            if func[e] < lower:
                                func[e] = lower
                            elif func[e] > upper:
                                func[e] = upper

        if isinstance(self.args.active, np.ndarray):
            nx, ny, nz = self.args.num_of_x, self.args.num_of_y, self.args.num_of_z
            active = self.args.active.reshape(nz, nx, ny)[0].astype(bool) & self.args.active.reshape(nz, nx, ny)[1].astype(bool)
            y_t, x_t = np.where(active)
            y_n, x_n = np.where(~active)
            t = [(y + 1, x + 1) for y, x in zip(y_t, x_t)]
            n = [(y + 1, x + 1) for y, x in zip(y_n, x_n)]
            loc = [(self.location['y'], self.location['x'])]

            d = []
            d_loc = []
            new = []
            for idx, l in enumerate(loc):
                if l in n:
                    d.append(l)
                    d_loc.append(idx)

            dist_mat = squareform(pdist(d + t))[:len(d), len(d):]
            nearest = np.min(dist_mat, axis=1)
            for i in range(len(d)):
                new.append(t[np.where(nearest[i] == dist_mat[i, :])[0][0]])

            for l_new, d_new in zip(d_loc, new):
                self.location['y'], self.location['x'] = d_new


        self.__round_location__()
        self.__rearrange_type__()

    def __rearrange_type__(self):
        type = self.type['type']

        for label, boundary in self.type['boundary'].items():
            if max(max(self.type['boundary'].values())) != max(boundary):
                if max(boundary) > type >= min(boundary):
                    self.type['label'] = label
                    self.type['index'] = self.well_types[label]
                    break
            else:
                if max(boundary) >= type >= min(boundary):
                    self.type['label'] = label
                    self.type['index'] = self.well_types[label]
                    break

    def __round_location__(self):
        for elem in self.attributes['location']:
            self.location[elem] = int(self.location[elem])


class PositionExample(Simulate):

    def __init__(self,
                 args,
                 wset=None,
                 location_fix=False,
                 type_fix=None,
                 drilling_time_fix=False,
                 control_fix=False,
                 well_type=None,
                 violation=None,
                 violation_check=True,
                 num_of_wells=None,
                 sampling_setting=None
                 ):
        super(PositionExample, self).__init__(args)
        self.args = args
        self.sampling_setting = sampling_setting
        if isinstance(args.active, np.ndarray):
            self.active = args.active
        else:
            self.active = None
        # maximum possible number of wells
        self.num_of_wells = num_of_wells
        self.num_of_max_wells = self.sampling_setting['Number of max wells']
        self.available_method = args.available_method

        # number of grids in (x, y, z)-direction
        self.num_of_x, self.num_of_y, self.num_of_z = args.num_of_x, args.num_of_y, args.num_of_z
        self.length_of_x, self.length_of_y, self.length_of_z = args.length_of_x, args.length_of_y, args.length_of_z
        self.num_of_grid = args.num_of_grid
        # 2023-04-06 re-defining in terms of radius(As far as possible)
        self.default_radius = (43560 * self.args.independent_area / pi) ** 0.5
        if sampling_setting['Default radius']:
            self.radius = self.default_radius
        else:
            layer1 = self.active.reshape(self.num_of_z, self.num_of_x, self.num_of_y)[0, :, :]
            layer2 = self.active.reshape(self.num_of_z, self.num_of_x, self.num_of_y)[1, :, :]
            possible_cell = (layer1 * layer2).astype(bool)
            self.radius = ((self.num_of_x*self.length_of_x * self.num_of_y*self.length_of_y * sum(sum(possible_cell))
                            /self.num_of_grid) / (self.num_of_max_wells * pi))**0.5

        if well_type is None:
            self.well_type = {'P': 1, 'No': 0, 'I': -1}  # -1: inj, 0: no well: 1: prod
        else:
            self.well_type = well_type

        self.type_fix = type_fix
        self.location_fix = location_fix
        self.drilling_time_fix = drilling_time_fix
        self.control_fix = control_fix
        self.wset = {'P': wset[:2], 'I': wset[2:], 'No':[0,0]}
        self.violation = violation
        self.violation_check = violation_check
    def initialize(self, ratio_of_infeasible=0.3, location=None, type_real=None, drilling_time=None, control=None,duplicate=None):
        if not duplicate:
            duplicate = [0]
        """
        :param ratio_of_infeasible: infeasible means not satisfying defined constraints
        :return: randomly initialized candidate solutions
        """

        num_of_x, num_of_y, num_of_z = self.num_of_x, self.num_of_y, self.num_of_z
        num_of_wells = self.num_of_wells
        num_of_max_wells = self.num_of_max_wells

        if isinstance(self.active, np.ndarray):
            layer1 = self.active.reshape(args.num_of_z, args.num_of_x, args.num_of_y)[0, :, :]
            layer2 = self.active.reshape(args.num_of_z, args.num_of_x, args.num_of_y)[1, :, :]
            possible_cell = (layer1 * layer2).astype(bool)
            self.possible_index = np.where(possible_cell.flatten())[0] + 1
            candidates = self.possible_index
        else:
            candidates = list(range(1, num_of_x * num_of_y * num_of_z + 1))

        # allow 30% of candidates infeasible
        # threshold = 0.001 if random.random() < ratio_of_infeasible else 0.5 * self.radius ** 2
        threshold = 0.001 if random.random() < ratio_of_infeasible else 0.5 * (self.default_radius ** 2)
        while True:
            position = []
            # random for type
            type_real_ = type_real
            if not self.type_fix:
                if not type_real:
                    type_real_ = self._type_real()
                    #type_real_.sort(reverse=True)
                    #n_p = type_real_.count(1)
                    #n_i = type_real_.count(-1)
                    for _ in range(num_of_max_wells - len(type_real_)):
                        type_real_.append(0)

            # random for location
            location_ = location
            if not self.location_fix:  # consider location as opt. variable
                if not location:  # location is not provided
                    location_ = []
                    if isinstance(self.sampling_setting['Quality'], dict):
                        for idx, k in enumerate(type_real_):
                            if k == 1:
                                if idx == 0:
                                    remain = candidates
                                # location_ += np.random.choice(remain, size=1,
                                #                              p=add_eps(softmax(np.log(self.sampling_setting['Quality']['P']
                                #                                                       [remain]))),
                                #                              replace=False).tolist()
                                location_ += np.random.choice(remain, size=1,
                                                             p=add_eps(softmax(consider_prob(add_eps(self.sampling_setting['Quality']['P']
                                                                                      [remain])))),
                                                             replace=False).tolist()

                                remain = list(set(candidates) - set(location_))
                            elif k == -1:
                                if idx == 0:
                                    remain = candidates
                                # location_ += np.random.choice(remain, size=1,
                                #                              p=add_eps(softmax(np.log(self.sampling_setting['Quality']['I']
                                #                                                       [remain]))),
                                #                              replace=False).tolist()
                                location_.append(np.random.choice(remain, size=1, p=add_eps(
                                    consider_prob(add_eps(self.sampling_setting['Quality']['I'][remain]))),
                                                                  replace=False))
                                remain = list(set(candidates) - set(location_))
                            else:
                                if idx == 0:
                                    remain = candidates
                                location_ += np.random.choice(remain,
                                                             size=1, replace=False).tolist()
                                remain = list(set(candidates) - set(location_))

                    elif self.sampling_setting['Use Quality']:
                        for idx, k in enumerate(type_real_):
                            if k == 1:
                                if idx == 0:
                                    remain = candidates
                                location_ += np.random.choice(remain, size=1,
                                                             p=add_eps(softmax(np.log(self.sampling_setting['Quality']['P']
                                                                                      [candidates]))),
                                                             replace=False).tolist()

                                remain = list(set(candidates) - set(location_))
                            else:
                                if idx == 0:
                                    remain = candidates
                                location_ += np.random.choice(remain, size=1, replace=False).tolist()
                                remain = list(set(candidates) - set(location_))
                    else:
                        location_ = np.random.choice(candidates, size=self.num_of_max_wells, replace=False).tolist()

            # type_real_ = type_real
            # if not self.type_fix:
            #     if not type_real:
            #         type_real_ = self._type_real()
            #         type_real_.sort(reverse=True)
            #         n_p = type_real_.count(1)
            #         n_i = type_real_.count(-1)
            #         for _ in range(num_of_max_wells - len(type_real_)):
            #             type_real_.append(0)
            # # random for location
            # location_ = location
            # if not self.location_fix:  # consider location as opt. variable
            #     if not location:  # location is not provided
            #         location_ = []
            #         if isinstance(self.sampling_setting['Quality'], dict):
            #             location_ += np.random.choice(candidates, size=n_p,
            #                                          p=add_eps(softmax(np.log(self.sampling_setting['Quality']['P']
            #                                                                   [candidates]))),
            #                                          replace=False).tolist()
            #             remain = list(set(candidates) - set(location_))
            #             location_ += np.random.choice(remain, size=n_i,
            #                                          p=add_eps(softmax(np.log(self.sampling_setting['Quality']['I']
            #                                                                   [remain]))),
            #                                          replace=False).tolist()
            #
            #             remain = list(set(candidates) - set(location_))
            #             location_ += np.random.choice(remain,
            #                                          size=self.num_of_max_wells - num_of_wells,
            #                                          replace=False).tolist()
            #
            #         elif self.sampling_setting['Use Quality']:
            #             location_ += np.random.choice(candidates, size=n_p,
            #                                          p=add_eps(softmax(np.log(self.sampling_setting['Quality']['P']
            #                                                                   [candidates]))),
            #                                          replace=False).tolist()
            #
            #             remain = list(set(candidates) - set(location_))
            #             location_ += np.random.choice(remain,
            #                         size = self.num_of_max_wells - n_p,
            #                                replace = False).tolist()
            #         else:
            #             location_ = np.random.choice(candidates, size=self.num_of_max_wells, replace=False).tolist()


            # random for drilling_time
            drilling_time_ = drilling_time
            if not self.drilling_time_fix:
                if not drilling_time:
                    drilling_time_ = ((ceil(num_of_max_wells/self.args.num_of_rigs)
                                       / (self.args.production_time/self.args.dstep))
                                      * np.random.rand(num_of_max_wells)).tolist()

            # random for well control
            control_ = control
            if not self.control_fix:
                if not control:
                    control_ = np.random.rand(num_of_max_wells, 1)
                    control_ = np.append(control_, pi * np.random.rand(num_of_max_wells, 1), axis=1)
                    control_ = np.append(control_, pi * np.random.rand(num_of_max_wells, 1), axis=1).tolist()

            for idx in range(num_of_max_wells):
                l = location_[idx] if location_ else None
                t = self._convert_type_value_to_str(type_real_[idx]) if type_real_ else None
                dt = drilling_time_[idx] if drilling_time_ else None
                wc = control_[idx] if control_ else None
                w = WellExample(self.args, location=l, type=t, drilling_time=dt,
                                well_control=wc, well_types=self.well_type)
                position.append(w)

            violation = self.violate(position, self.violation_check)
            if (violation <= threshold) & (not [sorted(location_), sorted(type_real_)] in duplicate):
                self.violation = violation
                break
        self.loc = location_
        self.t = type_real_
        self.wells = position

    def _convert_type_value_to_str(self, type):
        if type == 1:
            return 'P'
        elif type == -1:
            return 'I'
        else:
            return 'No'

    def _type_real(self):
        if self.type_fix:
            return self.type_real
        elif self.sampling_setting['Method'] in self.available_method:
            return [1] + [random.choice([1, -1]) for _ in range(self.num_of_wells - 1)]
        else:
            return [1] + [random.choice(list(self.well_type.values())) for _ in range(self.num_of_wells - 1)]

    def _make_type_variables(self, boundary, type_real):
        return [boundary[t][0] + (boundary[t][1] - boundary[t][0]) * random.random() for t in type_real]

    def violate(self, position, distance_check=True):
        args = self.args

        length_of_x, length_of_y = self.length_of_x, self.length_of_y
        location = [(well.location['x'], well.location['y']) for well in position if well.location]
        type_real = [well.type['label'] for well in position if well.type]

        if isinstance(self.active, np.ndarray):
            layer1 = self.active.reshape(self.num_of_z, self.num_of_x, self.num_of_y)[0, :, :]
            layer2 = self.active.reshape(self.num_of_z, self.num_of_x, self.num_of_y)[1, :, :]
            possible_cell = (layer1 * layer2).astype(bool)

            y, x = np.where(~possible_cell)
            x += 1
            y += 1
            boundaries = [(x_ * length_of_x, y_ * length_of_y) for x_, y_ in zip(x, y)]
        else:
            b = position[0].location['boundary']
            boundaries = dict()
            for length, coord in zip([length_of_x, length_of_y], ['x', 'y']):
                boundaries[coord] = [length * (b[coord][0] - 1), length * (b[coord][1] + 1)]

        coord_real = [(length_of_x * x, length_of_y * y) for (x, y) in location]

        drilled_well = []
        for idx, well in enumerate(coord_real):
            if type_real[idx] != 'No':
                drilled_well.append(coord_real[idx])

        production_well = []
        for idx, well in enumerate(coord_real):
            if type_real[idx] == 'P':
                production_well.append(coord_real[idx])

        violation = []
        if distance_check:
            # calculate distance between production wells and the other wells
            for idx1 in range(len(drilled_well)):
                for idx2 in range(idx1 + 1, len(drilled_well)):
                    violation.append(self.radius ** 2 - self._calculate_distance(drilled_well[idx1], drilled_well[idx2]))

            # calculate distance from the reservoir boundary
            if isinstance(self.active, np.ndarray):
                violation += list(self.default_radius ** 2 - self._calculate_boundary_distance_mod(production_well, boundaries))
            else:
                for p in production_well:
                    violation.append(self.default_radius ** 2 - self._calculate_boundary_distance(boundaries, p))

        # drilling_time = [ceil(well.drilling_time['time']*ceil(len(position)/args.num_of_rigs))
        #                  for well in position if well.drilling_time]
        drilling_time = [ceil(well.drilling_time['time'] * (args.production_time/args.dstep))
                         for well in position if well.drilling_time]
        if drilling_time:
            # check the existence of a well at the first time step
            production_well_idx = []
            for idx, type in enumerate(type_real):
                if type == 'P':
                    production_well_idx.append(idx)
            if 1 not in list(np.array(drilling_time)[production_well_idx]):
                violation.append(self.radius ** 2)
            # check whether the number of wells at once over the number of rigs
            else:
                s = sorted(drilling_time)[0]
                count = 0
                vio = 0
                for elem in sorted(drilling_time):
                    if elem == s:
                        count += 1
                    else:
                        s = elem
                        count = 1
                    if count > 3:
                        vio += 1
                violation.append(int(vio > 0) * self.radius ** 2)

        violation = [0 if vio < 0 else vio for vio in violation]

        return sum(violation)

    def _calculate_distance(self, p: tuple, q: tuple):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    def _calculate_boundary_distance(self, b: dict, p: tuple):
        distance = np.inf
        for p_elem, boundaries in zip(p, b.values()):
            if min([abs(p_elem - b) for b in boundaries]) < distance:
                distance = min([abs(p_elem - b) for b in boundaries])

        return distance ** 2

    def _calculate_boundary_distance_mod(self, prod, bound):
        return np.min(cdist(prod + bound, prod + bound)[:len(prod), len(prod):], axis=1) ** 2
    # def violate(self, position, distance_check=True):
    #     args = self.args
    #
    #     length_of_x, length_of_y = self.length_of_x, self.length_of_y
    #     location = [(well.location['x'], well.location['y']) for well in position if well.location]
    #     type_real = [well.type['label'] for well in position if well.type]
    #
    #     b = position[0].location['boundary']
    #     boundaries = dict()
    #     for length, coord in zip([length_of_x, length_of_y], ['x', 'y']):
    #         boundaries[coord] = [length * (b[coord][0] - 1), length * (b[coord][1] + 1)]
    #
    #     coord_real = [(length_of_x * x, length_of_y * y) for (x, y) in location]
    #
    #     # 2023-05-02 calculate to scipy(numpy)
    #     drilled_well = [coord_real[idx] for idx, well in enumerate(coord_real) if type_real[idx] != 'No']
    #     production_well = [coord_real[idx] for idx, well in enumerate(coord_real) if type_real[idx] == 'P']
    #     if distance_check:
    #         violation = np.power(self.radius, 2) - self._calculate_distance(drilled_well)  # h1
    #         violation = np.append(violation, (43560 * self.args.independent_area / pi)
    #                               - self._calculate_boundary_distance(boundaries, production_well))  # h2
    #
    #     drilling_time = [ceil(well.drilling_time['time'] * (args.production_time / args.dstep))
    #                      for well in position if well.drilling_time]
    #     vio = []
    #     if drilling_time:
    #         # check the existence of a well at the first time step
    #         production_well_idx = []
    #         for idx, type in enumerate(type_real):
    #             if type == 'P':
    #                 production_well_idx.append(idx)
    #         if 1 not in list(np.array(drilling_time)[production_well_idx]):
    #             vio.append(self.radius ** 2)
    #         # check whether the number of wells at once over the number of rigs
    #         else:
    #             s = sorted(drilling_time)[0]
    #             count = 0
    #             vio = 0
    #             for elem in sorted(drilling_time):
    #                 if elem == s:
    #                     count += 1
    #                 else:
    #                     s = elem
    #                     count = 1
    #                 if count > 3:
    #                     vio += 1
    #             vio.append(int(vio > 0) * self.radius ** 2)
    #
    #         violation = np.append(violation, vio) # h3
    #     return np.sum(violation[violation > 0])
    #
    # # 2023-05-02
    # def _calculate_distance(self, loc: list):
    #     return np.power(pdist(loc), 2)
    #
    # # def _calculate_boundary_distance(self, b: dict, p: tuple):
    # #     distance = np.inf
    # #     for p_elem, boundaries in zip(p, b.values()):
    # #         if min([abs(p_elem - b) for b in boundaries]) < distance:
    # #             distance = min([abs(p_elem - b) for b in boundaries])
    # def _calculate_boundary_distance(self, b: dict, p: list):
    #     coord = {'x': [], 'y': []}
    #     for x_, y_ in p:
    #         coord['x'].append(x_)
    #         coord['y'].append(y_)
    #     b_product = list(product(coord['x'], b['x'])) + list(product(b['y'], coord['y']))
    #     distance = squareform(pdist(p + b_product))[:len(p), len(p):]
    #     return np.power(np.min(distance, axis=1), 2)
    #
    # def _cut_boundary(self):
    #     for well in self.wells:
    #         well._cut_boundaries()

