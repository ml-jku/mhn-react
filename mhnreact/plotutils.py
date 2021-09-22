# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

Plot utils
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

plt.style.use('default')


def normal_approx_interval(p_hat, n, z=1.96):
    """ approximating the distribution of error about a binomially-distributed observation, {\hat {p)), with a normal distribution
    z = 1.96 --> alpha =0.05
    z = 1 --> std
    https://www.wikiwand.com/en/Binomial_proportion_confidence_interval"""
    return z*((p_hat*(1-p_hat))/n)**(1/2)


our_colors = {
    "lightblue": (  0/255, 132/255, 187/255),
    "red":       (217/255,  92/255,  76/255),
    "blue":      (  0/255, 132/255, 187/255),
    "green":     ( 91/255, 167/255,  85/255),
    "yellow":    (241/255, 188/255,  63/255),
    "cyan":      ( 79/255, 176/255, 191/255),
    "grey":      (125/255, 130/255, 140/255),
    "lightgreen":(191/255, 206/255,  82/255),
    "violett":   (174/255,  97/255, 157/255),
}


def plot_std(p_hats, n_samples,z=1.96, color=our_colors['red'], alpha=0.2, xs=None):
    p_hats = np.array(p_hats)
    stds = np.array([normal_approx_interval(p_hats[ii], n_samples[ii], z=z) for ii in range(len(p_hats))])
    xs = range(len(p_hats)) if xs is None else xs
    plt.fill_between(xs, p_hats-(stds), p_hats+stds, color=color, alpha=alpha)
    #plt.errorbar(range(13), asdf, [normal_approx_interval(asdf[ii], n_samples[ii], z=z) for ii in range(len(asdf))],
    #             c=our_colors['red'], linestyle='None', marker='.', ecolor=our_colors['red'])


def plot_loss(hist):
    plt.plot(hist['step'], hist['loss'] )
    plt.plot(hist['steps_valid'], np.array(hist['loss_valid']))
    plt.legend(['train','validation'])
    plt.xlabel('update-step')
    plt.ylabel('loss (categorical-crossentropy-loss)')


def plot_topk(hist, sets=['train', 'valid', 'test'], with_last = 2):
    ks = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
    baseline_val_res = {1:0.4061, 10:0.6827, 50: 0.7883, 100:0.8400}
    plt.plot(list(baseline_val_res.keys()), list(baseline_val_res.values()), 'k.--')
    for i in range(1,with_last):
        for s in sets:
            plt.plot(ks, [hist[f't{k}_acc_{s}'][-i] for k in ks],'.--', alpha=1/i)
    plt.xlabel('top-k')
    plt.ylabel('Accuracy')
    plt.legend(sets)
    plt.title('Hopfield-NN')
    plt.ylim([-0.02,1])


def plot_nte(hist, dataset='Sm', last_cpt=1, include_bar=True, model_legend='MHN (ours)',
             draw_std=True, z=1.96, n_samples=None, group_by_template_fp=False, schwaller_hist=None, fortunato_hist=None): #1.96 for 95%CI
    markers = ['.']*4#['1','2','3','4']#['8','P','p','*']
    lw = 2
    ms = 8
    k = 100
    ntes = range(13)
    if dataset=='Sm':
        basel_values = [0.        , 0.38424785, 0.66807858, 0.7916149 , 0.9051132 ,
       0.92531258, 0.87295875, 0.94865587, 0.91830721, 0.95993717,
       0.97215858, 0.9896713 , 0.99917817] #old basel_values = [0.0, 0.3882, 0.674, 0.7925, 0.9023, 0.9272, 0.874, 0.947, 0.9185, 0.959, 0.9717, 0.9927, 1.0]
        pretr_values = [0.08439423, 0.70743412, 0.85555528, 0.95200267, 0.96513376,
       0.96976397, 0.98373613, 0.99960286, 0.98683919, 0.96684724,
       0.95907246, 0.9839079 , 0.98683919]# old [0.094, 0.711, 0.8584, 0.952, 0.9683, 0.9717, 0.988, 1.0, 1.0, 0.984, 0.9717, 1.0, 1.0]
        staticQK = [0.2096, 0.1992, 0.2291, 0.1787, 0.2301, 0.1753, 0.2142, 0.2693, 0.2651, 0.1786, 0.2834, 0.5366, 0.6636]
        if group_by_template_fp:
            staticQK = [0.2651, 0.2617, 0.261 , 0.2181, 0.2622, 0.2393, 0.2157, 0.2184, 0.2   , 0.225 , 0.2039, 0.4568, 0.5293]
    if dataset=='Lg':
        pretr_values = [0.03410448, 0.65397054, 0.7254572 , 0.78969294, 0.81329924,
       0.8651173 , 0.86775655, 0.8593128 , 0.88184124, 0.87764794,
       0.89734215, 0.93328846, 0.99531597]
        basel_values = [0.        , 0.62478044, 0.68784314, 0.75089511, 0.77044644,
       0.81229423, 0.82968149, 0.82965544, 0.83778338, 0.83049176,
       0.8662873 , 0.92308414, 1.00042408]
        #staticQK = [0.03638, 0.0339 , 0.03732, 0.03506, 0.03717, 0.0331 , 0.03003, 0.03613, 0.0304 , 0.02109, 0.0297 , 0.02632, 0.02217] # on 90k templates
        staticQK = [0.006416,0.00686, 0.00616, 0.00825, 0.005085,0.006718,0.01041, 0.0015335,0.006668,0.004673,0.001706,0.02551,0.04074]
    if dataset=='Golden':
        staticQK = [0]*13
        pretr_values = [0]*13
        basel_values = [0]*13

    if schwaller_hist:
        midx = np.argmin(schwaller_hist['loss_valid'])
        basel_values = ([schwaller_hist[f't100_acc_nte_{k}'][midx] for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, '>10', '>49']])
    if fortunato_hist:
        midx = np.argmin(fortunato_hist['loss_valid'])
        pretr_values = ([fortunato_hist[f't100_acc_nte_{k}'][midx] for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, '>10', '>49']])

    #hand_val = [0.0 , 0.4, 0.68, 0.79, 0.89, 0.91, 0.86, 0.9,0.88, 0.9, 0.93]


    if include_bar:
        if dataset=='Sm':
            if n_samples is None:
                n_samples = [610, 1699, 287, 180, 143, 105, 70, 48, 124, 86, 68, 2539, 1648]
                if group_by_template_fp:
                    n_samples = [460, 993, 433, 243, 183, 117, 102, 87, 110, 80, 103, 3048, 2203]
        if dataset=='Lg':
            if n_samples is None:
                n_samples = [18861, 32226, 4220, 2546, 1573, 1191, 865, 652, 1350, 642, 586, 11638, 4958] #new
                if group_by_template_fp:
                    n_samples = [13923, 17709, 7637, 4322, 2936, 2137, 1586, 1260, 1272, 1044, 829, 21695, 10559]
                        #[5169, 15904, 2814, 1853, 1238, 966, 766, 609, 1316, 664, 640, 30699, 21471]
                        #[13424,17246, 7681, 4332, 2844,2129,1698,1269, 1336,1067, 833, 22491, 11202] #grouped fp
        plt.bar(range(11+2), np.array(n_samples)/sum(n_samples[:-1]), alpha=0.4, color=our_colors['grey'])

    xti = [*[str(i) for i in range(11)], '>10', '>49']
    asdf = []
    for nte in xti:
        try:
            asdf.append( hist[f't{k}_acc_nte_{nte}'][-last_cpt])
        except:
            asdf.append(None)

    plt.plot(range(13), asdf,f'{markers[3]}--', markersize=ms,c=our_colors['red'], linewidth=lw,alpha=1)
    plt.plot(ntes, pretr_values,f'{markers[1]}--', c=our_colors['green'],
             linewidth=lw, alpha=1,markersize=ms) #old [0.08, 0.7, 0.85, 0.9, 0.91, 0.95, 0.98, 0.97,0.98, 1, 1]
    plt.plot(ntes, basel_values,f'{markers[0]}--',linewidth=lw,
             c=our_colors['blue'], markersize=ms,alpha=1)
    plt.plot(range(len(staticQK)), staticQK, f'{markers[2]}--',markersize=ms,c=our_colors['yellow'],linewidth=lw, alpha=1)

    plt.title(f'USPTO-{dataset}')
    plt.xlabel('number of training examples')
    plt.ylabel('top-100 test-accuracy')
    plt.legend([model_legend, 'Fortunato et al.','FNN baseline',"FPM baseline", #static${\\xi X}: \\dfrac{|{\\xi} \\cap {X}|}{|{X}|}$
                'test sample proportion'])

    if draw_std:
        alpha=0.2
        plot_std(asdf, n_samples, z=z, color=our_colors['red'], alpha=alpha)
        plot_std(pretr_values, n_samples, z=z, color=our_colors['green'], alpha=alpha)
        plot_std(basel_values, n_samples, z=z, color=our_colors['blue'], alpha=alpha)
        plot_std(staticQK, n_samples, z=z, color=our_colors['yellow'], alpha=alpha)


    plt.xticks(range(13),xti);
    plt.yticks(np.arange(0,1.05,0.1))
    plt.grid('on', alpha=0.3)