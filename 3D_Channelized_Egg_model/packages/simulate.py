import time
from math import *
from packages.utils import *
import os, subprocess, shutil
import pandas as pd

import multiprocessing as mp
from joblib import Parallel, delayed


class Simulate:
    def __init__(self, args):
        self.args = args
        self.filepath = args.filepath
        self.active = args.active
        self.simulation_directory = args.simulation_directory
        self.save_directory = args.save_directory
        self.ecl_filename = args.ecl_filename
        self.frs_filename = args.frs_filename
        self.perm_filename = args.perm_filename
        self.active_filename = args.active_filename
        self.position_filename = args.position_filename
        self.constraint_filename = args.constraint_filename

        if not os.path.exists(self.simulation_directory):
            os.mkdir(self.simulation_directory)
            shutil.copy(f'{self.filepath}/$convert.bat', self.simulation_directory)

    def _run_program(self, program, filename):
        # command = fr"C:\\ecl\\2009.1\\bin\\pc\\{program}.exe {filename} > NUL"
        command = fr"C:\\ecl\\2024.2\\bin\\pc_x86_64\\{program}.exe {filename} > NUL"
        os.chdir(self.simulation_directory)
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        os.chdir('../')

    def _run_converter(self, filename):
        os.chdir(self.simulation_directory)
        command = fr"$convert < {filename} > NUL"
        subprocess.run(command, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        os.chdir('../')

    def _get_datafile(self, filename):
        with open(f'{self.filepath}/{filename}.DATA', 'r') as f:
            lines = f.readlines()
        return lines

    # def _set_datafile(self, filename, rawdata, idx):
    #     parallel = self.args.parallel
    #     with open(f'{self.simulation_directory}/{filename}.DATA', 'w') as f:
    #         for line in rawdata:
    #             if self.active_filename in line:
    #                 if parallel:
    #                     f.write(f"'{self.active_filename}_{idx}.DATA' /\n")
    #                 else:
    #                     f.write(f"'{self.active_filename}.DATA' /\n")
    #
    #             if self.perm_filename in line:
    #                 if parallel:
    #                     f.write(f"'{self.perm_filename}_{idx}.DATA' /\n")
    #                 else:
    #                     f.write(f"'{self.perm_filename}.DATA' /\n")
    #             elif self.constraint_filename in line:
    #                 f.write(f"'{self.constraint_filename}_{idx}.DATA' /\n")
    #             elif self.position_filename in line:
    #                 f.write(f"'{self.position_filename}_{idx}.DATA' /\n")
    #             else:
    #                 f.write(line)
    #         f.write('\n')

    def _set_datafile(self, filename, rawdata, idx):
        parallel = self.args.parallel
        with open(f'{self.simulation_directory}/{filename}.DATA', 'w') as f:
            for line in rawdata:
                if self.active_filename in line:
                    f.write(f"'{self.active_filename}_{idx}.DATA' /\n")

                if self.perm_filename in line:
                    f.write(f"'{self.perm_filename}_{idx}.DATA' /\n")
                elif self.constraint_filename in line:
                    f.write(f"'{self.constraint_filename}_{idx}.DATA' /\n")
                elif self.position_filename in line:
                    f.write(f"'{self.position_filename}_{idx}.DATA' /\n")
                else:
                    f.write(line)
            f.write('\n')

    def _set_posfile(self, position, idx):
        wells = position.wells

        with open(f'{self.simulation_directory}/{self.position_filename}_{idx}.DATA', 'w') as f:
            f.write('\nWELSPECS\n')
            for idx, well in enumerate(wells):
                if well.type['label'] == 'P':
                    f.write(f"PROD{idx + 1} 1 {well.location['x']} {well.location['y']} 1* OIL /\n")
                elif well.type['label'] == 'I':
                    f.write(f"INJ{idx + 1} 1 {well.location['x']} {well.location['y']} 1* WATER /\n")
            f.write('/\n\n')

            f.write('COMPDAT\n')
            for idx, well in enumerate(wells):
                if well.type['label'] == 'P':
                    f.write(f"PROD{idx + 1} {well.location['x']} {well.location['y']} 1 2 OPEN 2* 0.2 1* 0 /\n")
                elif well.type['label'] == 'I':
                    f.write(f"INJ{idx + 1} {well.location['x']} {well.location['y']} 1 2 OPEN 2* 0.2 1* 0 /\n")
            f.write('/\n')

    def _set_constfile(self, position, idx, total_time, tstep, dstep=90):

        def __set_control__(control_range, variable, t):
            low, high = control_range
            if variable == 'type':          # relaxation
                if well.type['label'] == 'P':
                    boundary = well.type['boundary']['P']
                    control = high - (well.type['type'] - boundary[0]) / (boundary[1] - boundary[0]) * (high - low)
                elif well.type['label'] == 'I':
                    boundary = well.type['boundary']['I']
                    control = low + (-well.type['type'] - boundary[1]) / (boundary[1] - boundary[0]) * (high - low)
            elif variable == 'control':     # by Awotunde(2014)
                t /= (total_time/dstep)
                alpha, beta, gamma = well.well_control['alpha'], well.well_control['beta'], well.well_control['gamma']
                control = (high - low)/2 * alpha * (1+cos(beta*(1-t)+gamma)) + low
            return control

        def __set_schedule__(variable, t):
            if variable == 'type':
                drilling_time = 0
            elif variable == 'control':
                drilling_time = ceil(total_time/dstep * well.drilling_time['time'])
            return t >= drilling_time

        wells = position.wells
        wset = position.wset

        control_var_type = "control" if wells[0].well_control else "type"
        num_of_dstep = ceil(total_time/dstep) if control_var_type == 'control' else 1
        num_of_tstep = int(total_time/num_of_dstep/tstep)
        drilling_term = range(num_of_dstep)

        for well in wells:
            well._set_control(wset, drilling_term)
            well._set_schedule(drilling_term)
        with open(f'{self.simulation_directory}/{self.constraint_filename}_{idx}.DATA', 'w') as f:
            for t in drilling_term:
                f.write('\nWCONPROD\n')
                for idx, well in enumerate(wells):
                    if well.type['label'] == 'P':
                        if well.schedule[t] == 1:
                            f.write(f"PROD{idx + 1} OPEN BHP 3* 1* 1* {well.control[t]} /\n")
                        else:
                            f.write(f"PROD{idx + 1} SHUT BHP 3* 1* 1* {wset['P'][1]} /\n")
                f.write('/\n\n')
                f.write('WCONINJE\n')
                for idx, well in enumerate(wells):
                    if well.type['label'] == 'I':
                        if well.schedule[t] == 1:
                            f.write(f"INJ{idx + 1} WATER OPEN BHP 1* 1* {well.control[t]} /\n")
                        else:
                            f.write(
                                f"INJ{idx + 1} WATER SHUT BHP 1* 1* {wset['I'][0]} /\n")
                f.write('/\n\n')
                f.write('TSTEP\n')
                f.write(f"{int(num_of_tstep)}*{tstep} /\n")

    def _get_proddata(self, idx) -> pd.DataFrame:
        os.chdir(self.simulation_directory)
        data, head, well = [], [], []
        #metric, unit = [], []
        with open(f'{args.ecl_filename}_{idx}.RSM', 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'TIME' in line:
                head += line.split()
            #    metric = list(line)
            # if '*10**3' in line:
            #    unit = list(line)
            if 'PROD' in line or 'INJ' in line:
                well += line.split()

            try:
                # if (float(line.split()[0]) != 0) and (float(line.split()[0]) % self.args.tstep == 0):
                if float(line.split()[0]) % args.tstep == 0:
                    data.append([float(l) for l in line.split()])
            except:
                continue

        well = well[::-1]
        head_re = []
        for h in head:
            if 'WOPT' in h or 'WWIT' in h:
                h += f'_{well.pop()}'
            head_re.append(h)

        if len(head_re) > 10:
            len_data = int(len(data) / 2)
            data_merge = np.concatenate((np.array(data[:len_data]), np.array(data[len_data:])), axis=1)
            prod_df = pd.DataFrame(data_merge, columns=head_re)
            # prod_df = prod_df.loc[:, ~prod_df.T.duplicated()]
        else:
            prod_df = pd.DataFrame(data, columns=head_re)
            # prod_df = prod_df.loc[:, ~prod_df.T.duplicated()]

        os.chdir('../')

        return prod_df

    def _get_griddata(self, idx, tstep_idx, filename, data_type):
        nx, ny = self.args.num_of_x, self.args.num_of_y

        if isinstance(self.active, np.ndarray):
            num_grid = sum(args.active == 1)
        else:
            num_grid = args.num_of_grid

        with open(self.simulation_directory+'/'+filename+'_'+str(idx)+'.F'+'0'*(4-len(str(tstep_idx)))+str(tstep_idx)) as f:
            lines = f.readlines()

        lines_converted = []
        for line in lines:
            lines_converted.append([element.strip() for element in line.split()])

        condition = False
        data = []

        for line in lines_converted:
            if f"'{data_type}" in line or f"'{data_type}'" in line:
                condition = True
            elif condition:
                data += [float(l) for l in line]
            if len(data) == num_grid:
                break
        if isinstance(self.active, np.ndarray):
            grid = np.zeros(args.num_of_grid * args.num_of_z).flatten()
            grid[self.active == 1] = np.array(data)
            data = grid
        return data

    def _get_drilldata(self, wells):
        num_of_dstep = int(self.args.production_time/self.args.dstep)       # 3600/90 = 40
        num_of_tstep = int(self.args.production_time / self.args.tstep)     # 3600/30 = 120
        diff = int(num_of_tstep / num_of_dstep)                             # 240/80 = 3
        if wells[0].drilling_time:
            drilling_times = [(ceil(well.drilling_time['time']*num_of_dstep)-1)*diff for well in wells]
            drill_data = [drilling_times.count(t) for t in range(0, num_of_tstep)]
        else:
            drill_data = [sum([0 if well.type['label'] == 'No' else 1 for well in wells])] + [0]*(num_of_tstep-1)
        return drill_data

    def _del_file(self, directory, filename):
        os.remove(f'{directory}/{filename}')

    def _convert_frs_result(self, idx, filename, tstep_idx):
        result_file = filename+'_'+str(idx)+'.F'+'0'*(4-len(str(tstep_idx)))+str(tstep_idx)
        if os.path.exists(f'{self.simulation_directory}/{result_file}'):
            self._del_file(self.simulation_directory, result_file)
        with open(f'{self.args.simulation_directory}/Restart_converter.log', 'w') as f:
            f.write('U\n')
            f.write(f'{filename}_{idx}\n')
            f.write('1\n')
            f.write('1\n')
            f.write(f'{tstep_idx}\n')
            f.write(f'{tstep_idx}\n')
            f.write('N\n')
            f.write('N\n')

    def _cal_fitness(self, wells, drill_data, prod_data: pd.DataFrame, parameters=None, prices=None):
        if prices is None:
            prices = [60/0.159, -3/0.159, -5/0.159]
        if 'FWIT' in prod_data.columns:
            parameters = ['FOPT', 'FWPT', 'FWIT']
        else:
            prices = [60 / 0.159, -3 / 0.159]
            parameters = ['FOPT', 'FWPT']

        for param, pr in zip(parameters, prices):
            prod_data[f'{param}_diff'] = prod_data[param].shift(-1) - prod_data[param]
            prod_data[f'{param}_discounted'] = prod_data.apply(lambda x: (x[f'{param}_diff']/(1+self.args.discount_rate)
                                                                          **(self.args.observed_term*(x.name+1)
                                                                             /self.args.discount_term))*pr, axis=1)
        prod_data['Dwell'] = drill_data + [None]
        prod_data['Dwell_discounted'] = prod_data.apply(lambda x: (-self.args.drilling_cost*x['Dwell']
                                                                   /(1+self.args.discount_rate)
                                                                   **(self.args.observed_term*x.name
                                                                      /self.args.discount_term)), axis=1)

        prod_data = prod_data.filter(regex='discounted')

        fitness = prod_data.sum().sum()
        # fitness -= self.args.drilling_cost*sum([1 if 'P' in well.type.keys() or 'I' in well.type.keys() else 0
        #                                         for well in wells])

        return fitness

    def eclipse(self, idx, position, perm, mark=True):

        fit_ens = []
        tecl_ens = []
        prod_ens = []

        make_permfield(f'{self.simulation_directory}/{self.perm_filename}_{idx}.DATA', perm)
        make_activefield(f'{self.simulation_directory}/{self.active_filename}_{idx}.DATA', self.active, args.num_of_x)

        # make_permfield(f'{self.perm_filename}.DATA', perm)
        # make_activefield(f'{self.active_filename}.DATA', args.active, args.num_of_x)
        # shutil.copy(f'{self.perm_filename}.DATA', self.simulation_directory)
        # shutil.copy(f'{self.active_filename}.DATA', self.simulation_directory)

        datafile_raw = self._get_datafile(self.ecl_filename)
        self._set_datafile(f'{self.ecl_filename}_{idx}', datafile_raw, idx)

        self._set_posfile(position, idx)
        self._set_constfile(position, idx, self.args.production_time, self.args.tstep, self.args.dstep)

        t_s = time.time()
        self._run_program('eclipse', f'{self.ecl_filename}_{idx}')
        t_span = time.time() - t_s

        prod_data = self._get_proddata(idx)
        drill_data = self._get_drilldata(position.wells)
        fit = self._cal_fitness(position.wells, drill_data, prod_data, None,
                                [self.args.oil_price, self.args.disposal_cost, self.args.injection_cost])

        prod_ens.append(prod_data)
        fit_ens.append(fit)
        tecl_ens.append(t_span)

        self.prod_ens = prod_ens
        self.fit_ens = fit_ens
        self.tecl_ens = tecl_ens
        self.prod_data = sum(prod_ens)/len(prod_ens)
        if mark:
            fit = sum(fit_ens)/len(fit_ens)
        else:
            self.fit = sum(fit_ens) / len(fit_ens)
        self.tecl = sum(tecl_ens)/len(tecl_ens)

        return fit, self.tecl

    def frontsim(self, idx, position, perm):

        if len(perm) == 1:
            perm = perm[0]
        # 2022-12-26
        # To save Permeability in PositionExample Object
        self.perm = perm

        make_permfield(f'{self.simulation_directory}/{self.perm_filename}_{idx}.DATA', perm)
        make_activefield(f'{self.simulation_directory}/{self.active_filename}_{idx}.DATA', self.active, args.num_of_x)
        #
        # make_permfield(f'{self.perm_filename}.DATA', perm)
        # make_activefield(f'{self.active_filename}.DATA', args.active, args.num_of_x)
        # shutil.copy(f'{self.perm_filename}.DATA', self.simulation_directory)
        # shutil.copy(f'{self.active_filename}.DATA', self.simulation_directory)

        num_of_tstep = int(self.args.streamline_time / self.args.tstep)
        datafile_raw = self._get_datafile(self.frs_filename)
        self._set_datafile(f'{self.frs_filename}_{idx}', datafile_raw, idx)

        self._set_posfile(position, idx)
        self._set_constfile(position, idx, self.args.streamline_time, self.args.tstep, self.args.dstep)

        t_s = time.time()
        self._run_program('frontsim', f'{self.frs_filename}_{idx}')
        t_span = time.time() - t_s

        self._convert_frs_result(idx, f'{self.frs_filename}', num_of_tstep)
        self._run_converter('Restart_converter.log')

        TOF_beg = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'TIME_BEG')
        TOF_end = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'TIME_END')
        area_beg = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'ID_BEG')
        area_end = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'ID_END')
        Pressure = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'PRESSURE')
        Swat = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'SWAT')
        self.Dynamic = {'Pressure': Pressure,'Swat': Swat}
        self.tof = {"TOF_beg": TOF_beg, "TOF_end": TOF_end}
        self.area = {"area_beg": area_beg, "area_end": area_end}
        self.tfrs = t_span
        return self.tof, self.Dynamic, self.tfrs


# ########################################################################################################################
# # You can use this model if you want to parallel run the ecl & frs #####################################################
# # Last update : 18:39 27-Dec-2022 ######################################################################################
# ########################################################################################################################
    def eclipse_parallel(self, idx, position, perm):
        '''
        set & run the ecl files for parallel simulation
        '''
        # ######################################################
        # print(perm)
        # print(self.args.perm_filename)
        # ######################################################
        # 2023-01-02
        # To solve I/O-related error (We assumed that.), genenrate Permeability Data in "simulation" floder directly,
        # rather than generate and copy Permeability Data.
        # == 현 위치에서 permx 파일 생성해서 복사하는 것이 아닌, simulation 폴더 내에서 바로 생성하도록 변경
        make_permfield(f'{self.simulation_directory}/{self.perm_filename}_{idx}.DATA', perm)
        make_activefield(f'{self.simulation_directory}/{self.active_filename}_{idx}.DATA', self.active, args.num_of_x)
        # make_permfield(f'{self.args.perm_filename}.DATA', perm)
        # shutil.copy(f'{self.args.perm_filename}.DATA', self.args.simulation_directory)

        datafile_raw = self._get_datafile(self.ecl_filename)
        self._set_datafile(f'{self.ecl_filename}_{idx}', datafile_raw, idx)
        self._set_posfile(position, idx)
        self._set_constfile(position, idx, self.args.production_time, self.args.tstep, self.args.dstep)
        self._run_program('eclipse', f'{self.ecl_filename}_{idx}')


    def ecl_result(self, idx, position, mark=True):
        '''
        get production data & NPV for parallel simulation
        '''
        prod_data = self._get_proddata(idx)
        drill_data = self._get_drilldata(position.wells)
        fit = self._cal_fitness(position.wells, drill_data, prod_data, None,
                                    [self.args.oil_price, self.args.disposal_cost, self.args.injection_cost])
        self.prod_data = prod_data
        if mark:
            self.fit = fit
        return self.prod_data, fit

    def frontsim_parallel(self, idx, position, perm):
        '''
        set & run the frs files for parallel simulation
        '''
        if len(perm) == 1: perm = perm[0]
        # 2023-01-02
        # To solve I/O-related error (We assumed that.), generate Permeability Data in "simulation" folder directly,
        # rather than generate and copy Permeability Data.
        # == 현 위치에서 permx 파일 생성해서 복사하는 것이 아닌, simulation 폴더 내에서 바로 생성하도록 변경
        make_permfield(f'{self.simulation_directory}/{self.perm_filename}_{idx}.DATA', perm)
        make_activefield(f'{self.simulation_directory}/{self.active_filename}_{idx}.DATA', self.active, args.num_of_x)
        # make_permfield(f'{self.args.perm_filename}.DATA', perm)
        # shutil.copy(f'{self.args.perm_filename}.DATA', self.args.simulation_directory)

        datafile_raw = self._get_datafile(self.frs_filename)
        self._set_datafile(f'{self.frs_filename}_{idx}', datafile_raw, idx)
        self._set_posfile(position, idx)
        self._set_constfile(position, idx, self.args.streamline_time, self.args.tstep, self.args.dstep)
        self._run_program('frontsim', f'{self.frs_filename}_{idx}')

    def converter(self, idx):
        num_of_tstep = int(self.args.streamline_time / self.args.tstep)
        self._convert_frs_result(idx, f'{self.frs_filename}', num_of_tstep)
        self._run_converter('Restart_converter.log')

    def frs_result(self, idx):
        '''
        get TOF, Dynamic & permeability for parallel simulation
        '''
        num_of_tstep = int(self.args.streamline_time / self.args.tstep)
        TOF_beg = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'TIME_BEG')
        TOF_end = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'TIME_END')
        area_beg = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'ID_BEG')
        area_end = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'ID_END')
        Pressure = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'PRESSURE')
        Swat = self._get_griddata(idx, num_of_tstep, self.frs_filename, 'SWAT')
        self.Dynamic = {'Pressure': Pressure, 'Swat': Swat}
        self.tof = {"TOF_beg": TOF_beg, "TOF_end": TOF_end}
        self.area = {"area_beg": area_beg, "area_end": area_end}

        # 2022-12-26
        return self.tof, self.Dynamic