import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import colorsys
from commonlibs.drawing_tools.draw_tools import get_ax_obj
from commonlibs.drawing_tools.draw_tools import ncolors
def norm_colors(colors):
    """
    (a, b, c) -> ele in [0, 1]
    :param colors: [(a, b, c)]
    :return: 
    """
    r_c = []
    def norm(c):
        c = np.array(c)
        if np.max(c) == 0:
            return list(c)
        else:
            return list(c/np.max(c))

    for c in colors:
        c = norm(c)
        r_c.append(c)
    return r_c

def plot_compare(ax, x, y1, y2, l1='line1', l2='line2', c1=(1, 0, 0), c2=(0, 0, 1)):
    ax.plot(x, y1, '-o', label=l1, color=c1)
    ax.plot(x, y2, '-o', label=l2, color=c2)
    ax.legend()

def simple_plot_compare(save_path, x, y1, y2,
                        l1='line1', l2='line2', c1='b', c2='r'):
    fig_all, ax = get_ax_obj(save_path)
    plot_compare(ax, x, y1, y2, l1, l2, c1, c2)
    plt.savefig(save_path)

def plot_mul_compare(ax, x, ys, ls=None, cs=None, legend=True, title='DDD',
                     x_label='', y_label=''):
    """
    
    :param ax: 
    :param x: 
    :param ys: ys
    :param ls: labels, str
    :param cs: colors, RGB, float, in [0, 1]
    :return: 
    """
    if not ls:
        ls = ['%d' % i for i in range(len(ys))]
    if not cs:
        cs = ncolors(len(ys))
        cs = norm_colors(cs)
    if not (len(ys) == len(ls) == len(cs)):
        print(len(ys), len(ls), len(cs))
        print(ys, ls, cs)
    assert len(ys) == len(ls) == len(cs)
    for y, l, c in zip(ys, ls, cs):
        ax.plot(x, y, '-o', label=l, color=c)
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)


def simple_plot_mul_compare(save_path, x, ys, ls=None, cs=None, legend=True
                            , title='DDD',x_label='', y_label=''
                            ):
    fig_all, ax = get_ax_obj(save_path)
    plot_mul_compare(ax, x, ys, ls, cs, legend=legend,
                     title=title, x_label=x_label, y_label=y_label)
    plt.savefig(save_path)
    plt.close()

# just with error bars
def plot_mul_compare_ebar(ax, x, ys, yerrs,
                          ls=None, cs=None, legend=True, title='DDD',
                     x_label='', y_label=''):
    """

    :param ax: 
    :param x: M
    :param ys: ys N x M
    :param yerrs: y errrors: N x (M x 2) or N x (M x 1) or N x 1
    :param ls: labels, str
    :param cs: colors, RGB, float, in [0, 1]
    :return: 
    """
    if not ls:
        ls = ['%d' % i for i in range(len(ys))]
    if not cs:
        cs = ncolors(len(ys))
        cs = norm_colors(cs)
    if not (len(ys) == len(ls) == len(cs)):
        print(len(ys), len(ls), len(cs))
        print(ys, ls, cs)
    assert len(ys) == len(ls) == len(cs)
    for y, yerr, l, c \
            in zip(ys, yerrs, ls, cs):
        # yerr: 2 x M
        ax.errorbar(x, y, yerr=yerr, fmt='-o', ecolor=c, color=c,  label=l, elinewidth=2, capsize=4)
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

# just with error bars
# plt.errorbar(x,y,yerr=error_range,fmt='o:',ecolor='hotpink',
# 			elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
def simple_plot_mul_compare_ebar(save_path, x, ys, yerrs,
                                 ls=None, cs=None,
                                 legend=True
                            , title='DDD', x_label='', y_label=''
                            ):
    fig_all, ax = get_ax_obj(save_path)
    plot_mul_compare_ebar(ax, x, ys, yerrs, ls, cs, legend=legend,
                     title=title, x_label=x_label, y_label=y_label)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    x = [0, 1, 2]
    y = [3, 5, 6]
    y1 = [3, 5, 6]
    y2 = [4, 6, 2]
    simple_plot_compare('./test.png', x, y1, y2)
    simple_plot_mul_compare('./test_mul.png', x, [y1, y2], ['y1', 'y2'])
