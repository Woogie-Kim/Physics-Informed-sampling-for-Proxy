from scipy import io
import matplotlib.pyplot as plt
import os
import numpy as np
import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
from random import seed
import random
from math import *
import glob
import re
import pickle
from matplotlib.colors import LinearSegmentedColormap
from tensorboard.backend.event_processing import event_accumulator
from parsing import args
from copy import copy
from scipy.special import softmax
import torch
import torch.backends.cudnn as cudnn
import collections

typo = 'Times New Roman'
with open('./data/parula.pkl', 'rb') as f:
    cm_data = pickle.load(f)
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

def load_matfile(filename, name="PERMX"):
    data = io.loadmat(filename)
    return data[name].transpose().tolist()

def make_permfield(filename, perm):
    with open(filename, 'w') as f:
        f.write('PERMX\n')
        for p in perm:
            try:
                f.write(f'{p[0]}\n')
            except:
                f.write(f'{p}\n')
        f.write('/')

def make_activefield(filename, active, nx):
    with open(filename, 'w') as f:
        f.write('ACTNUM \n')
        for ps in active.reshape(-1, nx):
            for p in ps:
                f.write(f'{p} ')
            f.write('\n')
        f.write('/')

def draw_qualitymap(qualitys, fname=None):
    prod = quality(qualitys['P'], islog=False, smooth=False, view=False) * 100
    inj = quality(qualitys['I'], islog=False, smooth=False, view=False) * 100
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    img = ax1.imshow(prod.reshape(args.num_of_x, -1), cmap='jet')
    ax1.set_title('Producer', fontsize=20)
    cbar = fig.colorbar(img, ax=ax1, shrink=0.72)
    cbar.ax.set_ylabel('Probability (%)')
    tx = np.arange(9, args.num_of_x, 10)
    ty = np.arange(9, args.num_of_y, 10)
    ax1.set_xticks(tx, tx + 1)
    ax1.set_yticks(ty, ty + 1)

    ax2 = fig.add_subplot(1, 2, 2)
    img = ax2.imshow(inj.reshape(args.num_of_x, -1), cmap='jet')
    ax2.set_title('Injector', fontsize=20)
    cbar = fig.colorbar(img, ax=ax2, shrink=0.72)
    cbar.ax.set_ylabel('Probability (%)')

    tx = np.arange(9, args.num_of_x, 10)
    ty = np.arange(9, args.num_of_y, 10)
    ax1.set_xticks(tx, tx + 1)
    ax1.set_yticks(ty, ty + 1)
    ax2.set_xticks(tx, tx + 1)
    ax2.set_yticks(ty, ty + 1)

    plt.tight_layout()
    if fname:
        plt.savefig(f'./fig/{fname}.png')
    plt.show()

def NPV_histogram(sample_npv, seperated=None,bins=None, is_2stage=True, fname=None):
    plt.figure()
    plt.rcParams['figure.figsize'] = (5,4)
    plt.rcParams['figure.dpi'] = 300
    if is_2stage:
        plt.hist([sample_npv[:seperated], sample_npv[seperated:]], bins=bins,color=['k', [0, 0, 0.7]], label=['Uniform', '2stage'], edgecolor='black')
    else:
        plt.hist(sample_npv, bins=bins, label=['Random'], edgecolor='black', alpha=0.7)
    plt.legend()
    plt.xlabel('NPV (MM$)')
    plt.ylabel('Number')
    if fname:
        plt.savefig(f'./fig/{fname}.png')
    plt.show()
    plt.close()


# 2022-12-26
# For Quality Map Calculation
def calculate_boundary(coord, nx, ny, length_x=120, length_y=120):
    '''
    Reservoir boundary까지 거리를 계산하기 위한 메서드
    :param coord: (x,y)로 구성된 좌표 정보
    :param nx: x좌표 격자 수
    :param ny: y좌표 격자 수
    :param length_x: x좌표 격자별 길이
    :param length_y: y좌표 격자별 길이
    :return: reservoir boundary distance
    '''
    x, y = coord
    x_boundary = length_x * min(nx - x, abs(x))
    y_boundary = length_y * min(ny - y, abs(y))
    return min(x_boundary, y_boundary)
def wellplacement_multilayer(pos, perm=None, ponly=False, islog=True, ismean=False, issl=False, view=True, fname=None):
    xs, ys, ts = decompose_wp(pos)
    if not isinstance(pos, list):
        pos = [pos]
        ms = 10
    if isinstance(perm, np.ndarray):
        perm_copy = copy(perm)
    else:
        perm_copy = copy(pos[0].perm)

    active = pos[0].active
    perm_copy[active==0] = 0
    p = perm_copy.reshape(args.num_of_z, args.num_of_x, args.num_of_y)
    if islog:
        p = np.log(p)
    if issl:
        c = 'jet'
    else:
        c = parula_map
    plt.rcParams['font.family'] = typo

    if not ismean:
        plt.rcParams['figure.figsize'] = (4*args.num_of_z,4)
        plt.rcParams['figure.dpi'] = 150
        fig = plt.figure()

        for i in range(args.num_of_z):
            ax = fig.add_subplot(1, args.num_of_z, i+1)
            draw_wp(xs, ys, ts,ms=ms, ponly=ponly)

            img = ax.imshow(p[i, :,:], cmap=c)
            if issl:
                ax.set_xlim([0, args.num_of_y])
                ax.set_ylim([args.num_of_x, 0])
            else:
                if islog:
                    img.set_clim(0.75, 8.5)
                ax.set_xlim([-2, args.num_of_x + 2])
                ax.set_ylim([args.num_of_y + 2, -2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Layer {i+1}')


            if issl:
                cbar = fig.colorbar(img, ax=ax, shrink=0.72)
                cbar.ax.set_ylabel('TOF')
            else:
                cbar = fig.colorbar(img, ax=ax, shrink=0.72)
                cbar.ax.set_ylabel('ln (k)')
        plt.tight_layout()
    else:
        plt.rcParams['figure.figsize'] = (4, 4)
        plt.rcParams['figure.dpi'] = 150
        plt.tight_layout()
        fig = plt.figure()
        p = np.average(p, axis=0)
        ax = fig.add_subplot(1, 1, 1)
        img = ax.imshow(p, cmap=c)
        draw_wp(xs, ys, ts,ms=ms, ponly=ponly)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Mean')
        if issl:
            ax.set_xlim([0, args.num_of_x])
            ax.set_ylim([args.num_of_y, 0])
        else:
            ax.set_xlim([-2, args.num_of_x+2])
            ax.set_ylim([args.num_of_y+2, -2])

        if issl:
            cbar = fig.colorbar(img, ax=ax, shrink=0.72)
            cbar.ax.set_ylabel('TOF')
        else:
            cbar = fig.colorbar(img, ax=ax, shrink=0.72)
            cbar.ax.set_ylabel('ln (k)')
        plt.tight_layout()
    if fname:
        plt.savefig(f'./fig/{fname}.png')
    elif view:
        plt.show()

def draw_perm(perm, active, islog=True, ismean=False, issl=False, view=True, fname=None):
    perm_copy = copy(perm)
    perm_copy[active==0] = 0
    p = perm_copy.reshape(args.num_of_z, args.num_of_x, args.num_of_y)
    if islog:
        p = np.log(p)
    if issl:
        c = 'jet'
    else:
        c = parula_map
    plt.rcParams['font.family'] = typo

    if not ismean:
        plt.rcParams['figure.figsize'] = (4*args.num_of_z,4)
        plt.rcParams['figure.dpi'] = 150
        fig = plt.figure()

        for i in range(args.num_of_z):
            ax = fig.add_subplot(1, args.num_of_z, i+1)
            img = ax.imshow(p[i, :,:], cmap=c)
            if issl:
                ax.set_xlim([0, args.num_of_x])
                ax.set_ylim([args.num_of_y, 0])
            else:
                if islog:
                    img.set_clim(0.75, 8.5)
                ax.set_xlim([-2, 62])
                ax.set_ylim([62, -2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Layer {i+1}')


            if issl:
                cbar = fig.colorbar(img, ax=ax, shrink=0.72)
                cbar.ax.set_ylabel('TOF')
            else:
                cbar = fig.colorbar(img, ax=ax, shrink=0.72)
                cbar.ax.set_ylabel('ln (k)')
        plt.tight_layout()
    else:
        plt.rcParams['figure.figsize'] = (4, 4)
        plt.rcParams['figure.dpi'] = 150
        fig = plt.figure()
        p = np.average(p, axis=0)
        ax = fig.add_subplot(1, 1, 1)
        img = ax.imshow(p, cmap=c)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Mean')
        if issl:
            ax.set_xlim([0, args.num_of_x])
            ax.set_ylim([args.num_of_y, 0])
        else:
            ax.set_xlim([-2, args.num_of_x+2])
            ax.set_ylim([args.num_of_x+2, -2])

        if issl:
            cbar = fig.colorbar(img, ax=ax, shrink=0.72)
            cbar.ax.set_ylabel('TOF')
        else:
            cbar = fig.colorbar(img, ax=ax, shrink=0.72)
            cbar.ax.set_ylabel('ln (k)')
        plt.tight_layout()
    if fname:
        plt.savefig(f'./fig/{fname}.png')
        plt.close()
    elif view:
        plt.show()

def get_regression_fix(true, pred, fname=None):
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['figure.dpi'] = 300
    #min_ = min(min(true), min(pred))
    #max_ = max(max(true), max(pred))
    plt.scatter(true, pred, c='k', clip_on=False)
    plt.plot([0,1200],[0,1200], '--')
    plt.xlim([0,1200])
    plt.ylim([0,1200])
    plt.xlabel('True NPV (MM$)')
    plt.ylabel('Predicted NPV (MM$)')
    plt.title(f'R2: {r2_score(true,pred):.3f}',fontsize=30)
    plt.legend(['Data', 'Unit slope line'])
    plt.tight_layout()
    if fname:
        plt.savefig(f'./fig/{fname}.png')
        plt.close()
    else:
        plt.show()

def get_regression(true, pred, min_,max_,fname=None):
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['figure.dpi'] = 300
    plt.scatter(true, pred, c='k')
    plt.plot([0.6* min_,1.05* max_],[0.6* min_,1.05* max_], '--')
    plt.xlim([0.6* min_,1.05* max_])
    plt.ylim([0.6* min_,1.05* max_])
    plt.xlabel('True NPV (MM$)')
    plt.ylabel('Predicted NPV (MM$)')
    plt.title(f'R2: {r2_score(true,pred):.3f}',fontsize=30)
    plt.legend(['Data', 'Unit slope line'])
    plt.tight_layout()
    if fname:
        plt.savefig(f'{fname}.png')
        plt.close()
    else:
        plt.show()

def get_regression2(ecl, proxy, fname=None):
    e = ecl.to_numpy().reshape(-1,1)
    p = proxy.to_numpy().reshape(-1,1)
    min_ = min(e.min(), p.min())
    max_ = max(e.max(), p.max())
    plt.scatter(e, p, c='k')
    # plt.plot(e, reg.predict(e),c=[0.7, 0, 0])
    plt.plot([min_,max_],[min_,max_], '--')
    plt.xlim([min_,max_])
    plt.ylim([min_,max_])
    plt.xlabel('True')
    plt.ylabel('Proxy')
    plt.title(f'R2: {r2_score(e,p):.3f}',fontsize=30)
    plt.legend(['Data', 'Unit slope line'])
    if fname:
        plt.savefig(f'{fname}.png')
        plt.close()
    else:
        plt.show()

# 2022-12-26
# For Plotting and Saving Regression Result
def get_regress(Model, filename=None, show=None):
    real = Model.reals
    prediction = Model.predictions
    real = [r[0] / 1e6 for r in real]
    prediction = [p[0] / 1e6 for p in prediction]
    value_range = [0, 1.05 * max(max(real, prediction))]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    ax.scatter(real, prediction, s=6, c='k')
    ax.plot(value_range, value_range, color='r', linewidth=1.2)
    ax.set_aspect('equal', adjustable='box')
    plt.title(fr"R$^{2}$: {Model.metric['r2_score'][0]:.4f}", fontweight='bold', )
    plt.xlabel('True NPV (MM$)', fontname='Times New Roman')
    plt.ylabel('Predicted NPV (MM$)', fontname='Times New Roman')
    plt.xlim(value_range)
    plt.ylim(value_range)
    plt.locator_params(nbins=value_range[-1] // 100)

    if filename:
        file_path = os.path.join(args.train_model_saved_dir, args.train_model_figure_saved_dir)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        plt.savefig(os.path.join(file_path, filename) + '.png', facecolor='white')
    if show:
        plt.show()
    plt.close(fig)

def draw_samples(samples, what=1, fname=None):
    if what == 1:
        name = 'Producer'
    elif what == -1:
        name = 'Injector'
    idx = decompose_index(samples)
    _, _, zs = decompose_wp(samples)
    loc_p = list(collections.Counter(idx[zs==what]).keys())
    count_p = list(collections.Counter(idx[zs==what]).values())


    a = np.zeros(args.num_of_grid)
    a[loc_p] = count_p

    plt.rcParams['figure.figsize'] = (5, 4)
    plt.rcParams['figure.dpi'] = 150
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'{name}')
    tx = np.arange(9, args.num_of_x, 10)
    ty = np.arange(9, args.num_of_y, 10)
    ax.set_xticks(tx, tx+1)
    ax.set_yticks(ty, ty+1)
    img = ax.imshow(a.reshape(args.num_of_x,-1),cmap='jet')
    plt.xlabel('X')
    plt.ylabel('Y')
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel('Number')
    plt.tight_layout()

    if fname:
        plt.savefig(f'./fig/{fname}_{name}.png')
    plt.show()

# 2023-01-06
# For Plotting Quality Map Example
def draw_map(x, input=None):
    input_num = len(x)
    typo = 'Times New Roman'

    plt.figure()

    if input_num == 1:
        fig, ax = plt.subplots((1 + input_num) // 2, 2 - input_num % 2, figsize=(4, 8))
        ax.imshow(x[0], cmap='jet')
        if input: ax.set_title(input[0], fontname=typo, fontweight='bold')

    else:
        if (1 + input_num) // 2 == 1:
            fig, ax = plt.subplots((1 + input_num) // 2, 2 - input_num % 2, figsize=(8, 8))
            for i in range(input_num):
                ax[i % 2].imshow(x[i], cmap='jet')
                if input: ax[i % 2].set_title(input[i], fontname=typo, fontweight='bold')
        else:
            fig, ax = plt.subplots((1 + input_num) // 3, 3, figsize=(8, 8))
            for i in range(input_num):
                ax[i // 3, i % 3].imshow(x[i], cmap='jet')
                if input: ax[i // 3, i % 3].set_title(input[i], fontname=typo, fontweight='bold')

    # fig.colorbar(ax, ax=ax)
    plt.savefig('./summary/inputs.png')


# 2023-05-02 Well placement를 그려, sampling에 저장
def draw_wp(xs, ys, ts, ms=3,ponly=True, filename=None, view=True):
    x_p, y_p, x_i, y_i = xs[ts == 1]-1, ys[ts == 1]-1, xs[ts == -1]-1, ys[ts == -1]-1
    if not ponly:
        plt.plot(x_i, y_i, 'o', c='k', markersize=ms, label='Inj', clip_on=False)
    plt.plot(x_p, y_p, 'o', c=[0.8, 0, 0], markersize=ms, label='Prod', clip_on=False)
    plt.axis('square')

    plt.xlim([0, args.num_of_x])
    plt.ylim([args.num_of_x, 0])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='lower right')
    if filename:
        plt.savefig(f'./sampling/{filename}.png')

    # plt.clf() if not view else plt.show()

def index_to_coord(index):
    """
    :param index: well location index
    :return: without return. set (x,y,z) coordinates, boundaries
    """
    # 2023-05-09 indexing 들어가는 것 eclipse 방식으로 수정
    nx, ny, nz = args.num_of_x, args.num_of_y, args.num_of_z
    x = np.mod(index, nx) + ((np.mod(index, nx) == 0) * nx)
    y = ceil(index / nx)
    while y > ny:
        y -= ny
    z = ceil(index / nx / ny)
    return [x, y]

def decompose_index(pos):
    idxs=[]
    if not isinstance(pos, list):
        pos = [pos]
    for p in pos:
        for w in p.wells:
            idxs.append(w.location['index'])
    return np.array(idxs)

def decompose_wp(pos):
    xs = []
    ys = []
    ts = []
    if not isinstance(pos, list):
        pos = [pos]
    for p in pos:
        for w in p.wells:
            xs.append(w.location['x'])
            ys.append(w.location['y'])
            if w.type['label'] == 'P':
                wt = 1
            elif w.type['label'] == 'I':
                wt = -1
            else:
                wt = 0
            ts.append(wt)
    return np.array(xs), np.array(ys), np.array(ts)
# def decompose_wp(pos):
#     wp = []
#     ts = []
#     if not isinstance(pos, list):
#         pos = [pos]
#     for s in pos:
#         for index in s.loc:
#             wp.append(index_to_coord(index))
#         ts += s.t
#     xs = np.array(wp)[:, 0]
#     ys = np.array(wp)[:, 1]
#     return xs, ys, np.array(ts)

def draw_wpmap(pos):
    if hasattr(pos, 'perm'):
        plt.imshow(pos.perm.reshape(args.num_of_x, args.num_of_y), cmap='jet')
        plt.colorbar()
    for w in pos.wells:
        if w.type['label'] == 'P':
            c = [0, 0, 1]
        elif w.type['label'] == 'I':
            c = 'k'
        plt.plot(w.location['x'], w.location['y'], 'o', c=c, markersize=5)
        plt.axis('square')
        plt.xlim([0.5, args.num_of_x + 0.5])
        plt.ylim([0.5, args.num_of_y + 0.5])
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.show()

def wellplacement(pos, ponly=False, filename=None):
    xs, ys, ts = decompose_wp(pos)
    if not isinstance(pos, list):
        pos = [pos]
    ms = 10

    plt.figure()
    draw_perm(pos[0].perm, pos[0].active, view=False, ismean=True)
    draw_wp(xs, ys, ts, ms=ms, ponly=ponly)
    plt.tight_layout()
    if filename:
        plt.savefig(f'./fig/{filename}.png')
    plt.show()
    plt.close()

# 2023-05-11 2-stage sampling
def preprocess_tof(samples):
    for s in samples:
        tofi = np.array(s.tof['TOF_end'])
        crit = tofi==0
        loc = decompose_index(s)-1
        crit[loc] = False
        tofi[crit] = 10000
        s.tof['TOF_end'] = tofi


def get_CDF(positions, quantile, view=False, fname=None):
    well_type = positions[0].well_type.keys()
    well_array = positions[0].sampling_setting['Well array']
    # top n%
    th = np.quantile([s.fit for s in positions], quantile)

    # number
    ngrid = args.num_of_x * args.num_of_y
    nwell = np.zeros_like(well_array)
    nloc = {label: np.zeros(ngrid) for label in well_type}

    # fitness
    fwell = np.zeros_like(well_array)
    floc = {label: np.zeros(ngrid) for label in well_type}


    for pos in positions:
        if pos.fit >= th:
            tmp_nwell = 0
            for w in pos.wells:
                nloc[w.type['label']][w.location['index'] - 1] += 1
                floc[w.type['label']][w.location['index'] - 1] += pos.fit
                if w.type['label'] != 'No': tmp_nwell += 1
            nwell[tmp_nwell - well_array[0]] += 1
            fwell[tmp_nwell - well_array[0]] += pos.fit / 1e6

    q_npv = {label: floc[label] / (nloc[label] + 1e-10) for label in well_type}
    cdf_npv = {label: np.cumsum(q_npv[label] / np.sum(q_npv[label])) for label in well_type}
    pdf_npv ={label: add_eps((q_npv[label] / np.sum(q_npv[label]))) for label in well_type}

    # visualization
    if view:
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (8, 5)
        plt.rcParams['font.size'] = 12

        fig, ax = plt.subplots()
        ax.bar(args.array_of_wells, nwell)
        plt.xlim([args.array_of_wells[0] - 0.5, args.array_of_wells[-1] + 0.5])
        ax.set_ylabel('The number of samples', fontname=typo)
        ax.set_xlabel('Well number', fontname=typo)
        plt.show() if not fname else plt.savefig(f'./sampling/{fname}_bar.png')

        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (6, 5)
        plt.rcParams['font.size'] = 12

        fig, ax = plt.subplots()
        ax.plot(cdf_npv['P'], c=[0.7, 0, 0], label='P')
        ax.plot(cdf_npv['I'], c='k', label='I')
        plt.xlim([0, ngrid])
        plt.ylim([0, 1])
        ax.set_ylabel('CDF', fontname=typo)
        ax.set_xlabel('Well index', fontname=typo)
        plt.legend()
        plt.show() if not fname else plt.savefig(f'./sampling/{fname}_CDF.png')
    return pdf_npv, q_npv, nwell


def make_quality(permeability, view=False):
    permeability_q = copy(permeability)
    if np.min(permeability_q) <0:
        permeability_q -= np.min(permeability_q)
    p = sum(permeability_q)
    q = permeability_q / p
    if view:
        plt.imshow(q.reshape(args.num_of_x, args.num_of_x), cmap='jet')
        plt.show()
        plt.show()
    return q

def quality(permeability, islog=False, smooth=False, fname = None, projection = '3d',view=False):
    permeability_q = copy(permeability)
    nx = np.sqrt(len(permeability_q)).astype(int)
    if islog:
        perm_revised = np.log(permeability_q)
    else:
        perm_revised = permeability_q
    if smooth:
        perm_revised += np.average(perm_revised)

    q = make_quality(perm_revised)
    x, y = np.where(q.reshape(-1,args.num_of_x) == q.reshape(-1,args.num_of_y))
    if view:
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (6, 6)
        plt.rcParams['image.cmap'] ='jet'
        #plt.rcParams['font.size'] = 12
        fig = plt.figure()
        if projection == '3d':
            from scipy.ndimage.filters import gaussian_filter
            ax = fig.add_subplot(projection='3d')
            plot = ax.scatter(x, y, q, c=q)
            X, Y = np.meshgrid(x, y)
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_zlabel('Probability')
            # fig.colorbar(plot, shrink=0.5, aspect=5)
        elif projection == 'x':
            ax = fig.add_subplot()
            ax.scatter(x, q, c=q)
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Probability')
        elif projection == 'x_hist':
            plt.rcParams['figure.figsize'] = (7, 6)
            gs = fig.add_gridspec(1, 2, width_ratios=(4, 1),
                                  left=0.1, right=0.9,
                                  wspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
            scatter_hist(x, q, ax, ax_histy)
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Probability')
        elif projection == 'y':
            ax = fig.add_subplot()
            ax.scatter(y, q, c=q)
            ax.set_xlabel('Y coordinate')
            ax.set_ylabel('Probability')

        elif projection == 'xy':
            ax = fig.add_subplot()
            ax.imshow(q.reshape(-1, args.num_of_x))
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
        if fname:
            plt.tight_layout()
            plt.savefig(f'./fig/{fname}.png',bbox_inches='tight')
        plt.show()
    return q
def SLquality(perm, samples, setting):
    area_prod = np.zeros((len(perm), len(samples)))
    area_inj = np.zeros((len(perm), len(samples)))
    perm_prod = np.zeros((len(perm), len(samples)))
    perm_inj = np.zeros((len(perm), len(samples)))
    well_num = {i: [] for i in setting['Well array']}

    for k, sample in enumerate(samples):
        x, y, t = decompose_wp(sample)
        ids = decompose_index(sample)
        num = len(x[t != 0])
        tofp = copy(np.array(sample.tof['TOF_end']))
        tofi = copy(np.array(sample.tof['TOF_beg']))
        area_p = copy(np.array(sample.area['area_end']))
        area_i = copy(np.array(sample.area['area_beg']))

        plist = sorted(list(set(area_p)))
        ilist = sorted(list(set(area_i)))

        area_p[area_p == plist[-1]] = 0.0
        area_i[area_i == ilist[-1]] = 0.0

        tmp = []
        for i in plist[1:-1]:
            pp = copy(tofp)
            pp[area_p != i] = args.max_tof
            area_prod[ids[int(i) - 1] - 1, k] = sum(pp != args.max_tof)
            perm_prod[ids[int(i) - 1] - 1, k] = np.average(perm[pp != args.max_tof])
            tmp.append(sum(pp != args.max_tof) * np.average(perm[pp != args.max_tof]))
        for i in ilist[1:-1]:
            pi = copy(tofi)
            pi[area_i != i] = args.max_tof
            area_inj[ids[int(i) - 1] - 1, k] = sum(pi != args.max_tof)
            perm_inj[ids[int(i) - 1] - 1, k] = np.average(perm[pi != args.max_tof])
            tmp.append(sum(pi != args.max_tof) * np.average(perm[pi != args.max_tof]))

        if not num in well_num.keys():
            well_num[num] = []
        well_num[num].append(np.sum(tmp))

    for i in well_num.keys():
        well_num[i] = np.average(well_num[i])

    well_potential = [i for i in well_num.values()]
    p_well = softmax(np.log(well_potential))

    potential = {}
    quality = {}
    potential['P'] = area_prod[area_prod != 0] * perm_prod[area_prod != 0]
    potential['I'] = area_inj[area_inj != 0] * perm_inj[area_inj != 0]
    quality['P'] = np.zeros(args.num_of_grid)
    quality['I'] = np.zeros(args.num_of_grid)

    inj_loc = np.where(area_inj != 0)[0]
    prod_loc = np.where(area_prod != 0)[0]

    for i in list(set(prod_loc)):
        quality['P'][i] = np.average(potential['P'][prod_loc == i])

    for i in list(set(inj_loc)):
        quality['I'][i] = np.average(potential['I'][inj_loc == i])

    return quality, p_well, potential

# def SLquality(perm, samples, setting):
#     area_prod = np.zeros((len(perm), len(samples)))
#     area_inj = np.zeros((len(perm), len(samples)))
#     perm_prod = np.zeros((len(perm), len(samples)))
#     perm_inj = np.zeros((len(perm), len(samples)))
#     well_num = {i: [] for i in setting['Well array']}
#
#     for k, sample in enumerate(samples):
#         x, y, t = decompose_wp(sample)
#         ids = decompose_index(sample)
#         num = len(x[t != 0])
#         tofp = copy(np.array(sample.tof['TOF_end']))
#         tofi = copy(np.array(sample.tof['TOF_beg']))
#         area_p = copy(np.array(sample.area['area_end']))
#         area_i = copy(np.array(sample.area['area_beg']))
#
#         plist = sorted(list(set(area_p)))
#         ilist = sorted(list(set(area_i)))
#
#         area_p[area_p == plist[-1]] = 0.0
#         area_i[area_i == ilist[-1]] = 0.0
#
#         tmp = []
#         for i in plist[1:-1]:
#             pp = copy(tofp)
#             pp[area_p != i] = args.max_tof
#             area_prod[ids[int(i) - 1] - 1, k] = sum(pp != args.max_tof)
#             perm_prod[ids[int(i) - 1] - 1, k] = np.average(perm[pp != args.max_tof])
#             tmp.append(sum(pp != args.max_tof) * np.log(np.average(perm[pp != args.max_tof])))
#         for i in ilist[1:-1]:
#             pi = copy(tofi)
#             pi[area_i != i] = args.max_tof
#             area_inj[ids[int(i) - 1] - 1, k] = sum(pi != args.max_tof)
#             perm_inj[ids[int(i) - 1] - 1, k] = np.average(perm[pi != args.max_tof])
#             tmp.append(sum(pi != args.max_tof) * np.log(np.average(perm[pi != args.max_tof])))
#
#         if not num in well_num.keys():
#             well_num[num] = []
#         well_num[num].append(np.sum(tmp))
#
#     for i in well_num.keys():
#         well_num[i] = np.average(well_num[i])
#
#     well_potential = [i for i in well_num.values()]
#     p_well = softmax(np.log(well_potential))
#
#     potential = {}
#     quality = {}
#     potential['P'] = area_prod[area_prod != 0] * np.log(perm_prod[area_prod != 0])
#     potential['I'] = area_inj[area_inj != 0] * np.log(perm_inj[area_inj != 0])
#     quality['P'] = np.zeros(args.num_of_grid)
#     quality['I'] = np.zeros(args.num_of_grid)
#
#     inj_loc = np.where(area_inj != 0)[0]
#     prod_loc = np.where(area_prod != 0)[0]
#
#     for i in list(set(prod_loc)):
#         quality['P'][i] = np.average(potential['P'][prod_loc == i])
#
#     for i in list(set(inj_loc)):
#         quality['I'][i] = np.average(potential['I'][inj_loc == i])
#
#     return quality, p_well, potential

def scatter_hist(x, y, ax, ax_histy):
    # no labels
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, c=y)

    # now determine nice limits by hand:
    ymax =  np.max(np.abs(y))

    ybins = np.linspace(0, ymax, 20)
    n, bins, patches = ax_histy.hist(y, bins=ybins, orientation='horizontal')
    cm = plt.cm.get_cmap('RdYlBu_r')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))



# 2023-01-06
# For Naming Results with Combination List
def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()


def DrawNPV(Opt):
    typo = 'Times New Roman'
    NPV_gbest = []
    for gbest in Opt.algorithm.gbest:
        NPV_gbest.append(gbest.fit / 1e6)

    plt.figure()
    plt.plot(np.linspace(0, args.num_of_generations, args.num_of_generations + 1), NPV_gbest, marker='*', color='r',
             linewidth=1.2)
    plt.xlim([0, args.num_of_generations])
    plt.title("Optimization Results", fontname=typo, fontweight='bold', fontsize=20)
    plt.xlabel('Generations', fontname=typo)
    plt.ylabel('NPV (MM$)', fontname=typo)
    plt.xticks(np.linspace(0, args.num_of_generations, 5))
    plt.show()


def Calculate_ecdf(sample, percentile):
    sample_edf = edf.ECDF(sample)
    slope_changes = sorted(set(sample))
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)
    return inverted_edf(percentile)


def ViewParticles(Opt, other=True, n_components=3, lag=20, Npv_map=True):
    typo = 'Times New Roman'
    fig = plt.figure(figsize=(12, 12))
    iter = range(0, args.num_of_generations + 1, lag)
    components = [f'P{str(i + 1)}' for i in range(n_components)]
    if Npv_map and other:
        ax = fig.add_subplot(111)
    elif n_components == 3 or Npv_map:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    for i in iter:
        vector_list = []
        NPV = []
        for pbest in Opt.algorithm.positions_all[i]:
            NPV.append(pbest.fit / 1e6)
            x = [];
            y = [];
            type_ = []
            for w in pbest.wells:
                x.append(w.location['x'])
                y.append(w.location['y'])
                type_.append(w.type['index'])
            vector_list.append(x + y + type_)
        df = pd.DataFrame(vector_list)
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)
        df_scaled = scaler.fit_transform(df)
        df_pca = pca.fit_transform(df_scaled)
        df_pca = pd.DataFrame(df_pca, columns=components)
        color = [random.random(), random.random(), random.random()]

        # 3D Visualization
        if Npv_map and other:
            ax.scatter(df_pca['P1'], NPV, color=color)
        elif n_components == 3:
            ax.scatter(df_pca['P1'], df_pca['P2'], df_pca['P3'], color=color)
        elif Npv_map:
            ax.scatter(df_pca['P1'], df_pca['P2'], NPV, color=color)
        else:
            ax.scatter(df_pca['P1'], df_pca['P2'], color=color)
    plt.legend([f'Tstep{num}' for num in iter])

    if Npv_map and other:
        ax.set_xlabel('Component 1', fontname=typo)
        ax.set_ylabel('NPV (MM$)', fontname=typo)
    elif n_components == 3 or Npv_map:
        ax.set_xlabel('Component 1', fontname=typo)
        ax.set_ylabel('Component 2', fontname=typo)
        if n_components == 3:
            ax.set_zlabel('Component 3', fontname=typo)
        else:
            ax.set_zlabel('NPV (MM$)', fontname=typo)
    else:
        plt.xlabel('Component 1', fontname=typo)
        plt.ylabel('Component 2', fontname=typo)
    #plt.savefig(f'./training/{filename}')


def load_tf(dir_path):
    prefix = "logs/CNN/230201_Combination_test/"
    dirname = []
    inputs = []
    inputs_name = []
    dfs = []
    for dir in glob.glob(prefix + '/*'):
        dirname.append(dir + f'/{dir_path}')
        inputs.append([cat[1:-1] for cat in re.findall(r"('\w+')", dir)])
        inputs_name.append("_".join(inputs[-1]))

    for idx, dir in enumerate(dirname):
        ea = event_accumulator.EventAccumulator(dir, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        dframes = {}
        mnames = ea.Tags()['scalars']

        for n in mnames:
            # dframes[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", n.replace('val/', '')])
            dframes[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", inputs_name[idx]])
            dframes[n].drop("wall_time", axis=1, inplace=True)
            dframes[n] = dframes[n].set_index("epoch")
        dfs.append(pd.concat([v for k, v in dframes.items()], axis=1))
    return pd.concat(dfs, axis=1)


def df_log(data, input_list):
    dic = {}
    for input_1 in input_list:
        data_list = []
        for idx, key in enumerate(data.keys()):
            if input_1 in key:
                data_list.append(data.iloc[::, idx])
        cat = pd.concat(data_list, axis=1)
        dic[input_1] = {'total_log': cat, 'mean': cat.mean(axis=1)}
        input_list = input_list[1::]

        for input_2 in input_list:
            data_list = []
            for idx, key in enumerate(data.keys()):
                if input_1 in key and input_2 in key:
                    data_list.append(data.iloc[::, idx])
            cat = pd.concat(data_list, axis=1)
            dic['_'.join([input_1, input_2])] = {'total_log': cat, 'mean': cat.mean(axis=1)}
    return dic


def plot_log(dic, legend, filename, show=True, c=None):
    typo = 'Times New Roman'
    fig = plt.figure(figsize=(12, 6))
    for input_ in args.input_list:
        plt.plot(dic[input_]['mean'], c=c)
    plt.xlabel('epoch', fontname=typo, fontsize=20)
    plt.ylabel('RSME', fontname=typo, fontsize=20)
    plt.xlim(0, 20)
    plt.xticks(np.linspace(0, 20, 5))
    if show:
        plt.legend(legend, bbox_to_anchor=(0.85, 1), ncols=5)
        plt.savefig(f'./summary/{filename}.png')
        plt.show()


def setter(setting):
    if setting['Number of Samples'] >= len(setting['Well array']):
        rst = [int(setting['Number of Samples'] / len(setting['Well array'])) for _ in setting['Well array']]
        i = 0
        while sum(rst) != setting['Number of Samples']:
            rst[i] += 1
            i += 1
    else:
        for_rst = np.linspace(0, len(setting['Well array'])-1, setting['Number of Samples']).astype(int)
        rst = [0 for _ in setting['Well array']]
        for j in for_rst:
            rst[j] = 1
    return rst

def namer(setting, id, origin):
    if (setting['Method'] in ['2stage', 'physics']):
        return f"wo_{origin}_{setting['Method']}_{id}_egg.pkl"
    else:
        return f"wo_{setting['Number of Samples']}_{setting['Method']}_{id}_egg.pkl"

def draw_relative_error(wpo, fname=None):
    if not os.path.isdir('fig'):
        os.mkdir('fig')
    if not os.path.isdir('results'):
        os.mkdir('results')

    for best in ['gbest']:
        ecl = np.array(wpo.fits_true[best])
        proxy = np.array(wpo.fits_proxy[best])
        rel_error(ecl, proxy, f"{best}_{fname}")

def rel_error(ecl, proxy, fname=None):
    gen = list(range(len(proxy)))
    df = pd.DataFrame(columns=['Gen', 'ecl', 'proxy'])
    rel_error = (proxy - ecl) / ecl
    if rel_error.ndim == 2:
        rel_error = np.average(rel_error, axis=1)
        rst = {'gen':gen ,'ecl':ecl, 'proxy':proxy}
        if fname:
            with open(f'./results/{fname}.pkl', 'wb') as f:
                pickle.dump(rst, f)
    else:
        df['Gen'] = gen
        df['ecl'] = ecl
        df['proxy'] = proxy
        df.to_csv(f'./results/{fname}.csv')

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 2)
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots()
    ax.scatter(gen, rel_error, edgecolors='k', facecolors='w', clip_on=False)
    ax.axhline(y=0, color='r')
    plt.xlim([0, gen[-1]])
    plt.ylim([-0.2, 0.2])
    ax.set_ylabel('Relative error', fontname=typo)
    ax.set_xlabel('Generation', fontname=typo)
    if fname:
        plt.savefig(f'./fig/{fname}_err.png')

    else:
        plt.show()
    if rel_error.ndim == 1:
        return df

def add_eps(list_prob):
    if sum(list_prob) == 0.0:
        l = len(list_prob)
        list_prob = [1/l for _ in list_prob]
    eps = (1 - sum(list_prob))
    list_prob[np.argmax(list_prob)] += eps
    return list_prob

def consider_prob(list_prob):
    total_sum = sum(list_prob)
    normalized_list = [x / total_sum for x in list_prob]
    normalized_sum = sum(normalized_list)
    difference = 1 - normalized_sum
    if difference != 0:
        min_index = normalized_list.index(min(normalized_list))
        normalized_list[min_index] += difference

    # Ensure that the minimum value is not negative and set it to 0 if very small negative
    normalized_list = [max(0, x) for x in normalized_list]

    return normalized_list

def draw_graph(ecl, proxy,fname=None):
    plt.figure()
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.rcParams['font.size'] = 12
    gen = np.linspace(0, len(proxy) - 1, len(proxy))
    fig, ax = plt.subplots()
    ax.plot(gen, np.array(proxy) / 1e6, c=[0.8, 0, 0])
    ax.plot(gen, np.array(ecl) / 1e6, c=[0, 0, 0.8])
    # Add scatter points at every 20th generation
    scatter_indices = np.arange(0, len(gen), 20)
    ax.scatter(gen[scatter_indices], np.array(proxy)[scatter_indices] / 1e6, c=[0.8, 0, 0], marker='o', label='proxy'
               , clip_on=False)
    ax.scatter(gen[scatter_indices], np.array(ecl)[scatter_indices] / 1e6, c=[0, 0, 0.8], marker='x', label='ecl'
               , clip_on=False)
    ax.legend()
    ax.set_ylabel('NPV (MM$)', fontname=typo)
    ax.set_xlabel('Generation', fontname=typo)
    plt.xlim([0, gen[-1]])
    ax.set_ylim(0, 1300)

    plt.tight_layout()
    if fname:
        plt.savefig(f'./fig/{fname}.png')
        plt.clf()
    plt.show()

def draw_graph2(ecl, proxy,fname=None):
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 4)
    plt.rcParams['font.size'] = 12

    ecl = ecl.fits
    proxy_true = proxy.fits_true['gbest']
    proxy_pred = proxy.fits_proxy['gbest']
    gen = np.linspace(0, len(proxy_true) - 1, len(proxy_true))
    fig, ax = plt.subplots()
    ax.plot(gen, proxy_pred, linewidth=2.5, c=[0.8, 0, 0], label='proxy [pred]')
    ax.plot(gen[::10], proxy_pred[::10], 'o', linewidth=2.5, c=[0.8, 0, 0])
    ax.plot(gen, proxy_true, linewidth=2.5, c=[0, 0, 0.8], label='proxy [true]')
    ax.plot(gen[::10], proxy_true[::10], 'o', linewidth=2.5, c=[0, 0, 0.8])
    ax.plot(gen, ecl,linewidth=2.5, c='k', label='ecl')
    ax.plot(gen[::10], ecl[::10], 'o', linewidth=2.5, c='k')

    ax.legend()
    ax.set_ylabel('NPV ($)', fontname=typo)
    ax.set_xlabel('Generation', fontname=typo)
    plt.xlim([0, gen[-1]])
    if fname:
        plt.savefig(f'./fig/{fname}.png')
    plt.show()

def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def opt_sampling_setter(setting, opt_setting):
    setting['Quality'] = None
    setting['Method'] = None
    setting['Default radius'] = True
    setting['Infeasible ratio'] = 0.3
    setting['Number of Samples'] = opt_setting['Number of particles']
    setting['Sample array'] = setter(setting)
    return setting

def error_detect(sampling_setting):
    assert sampling_setting['Method'] != '2stage' or sampling_setting['Number of Samples'] > 16, \
        'Two-stage sampling method requires at least 16 samples for generating a quality map'

def error_detect_for_hypeopt(pars):
    assert pars.num_of_train_sample * (1 - args.train_ratio - args.validate_ratio) > 2 or not pars.use_bayesopt, \
        'Bayesian optimization requires at least 2 test data for calculating R2 as the objective function'
