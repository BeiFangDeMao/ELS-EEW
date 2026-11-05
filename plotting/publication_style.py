# -*- coding: utf-8 -*-
"""Shared SCI figure styling. All sizes are final printed points/mm."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ft2font import FT2Font

MM = 1/25.4
COLORS = {'train':'#0072B2', 'validation':'#D55E00', 'test':'#009E73'}
HATCHES = {'train':'', 'validation':'///', 'test':'...'}
LABELS = {
    'en': {'train':'Training', 'validation':'Validation', 'test':'Test',
           'mag':'Magnitude, $M$', 'dis':'Epicentral distance (km)',
           'frequency':'Sample proportion (%)', 'count':'Record count', 'azi':'Azimuth (°)',
           'rose_frequency':'Frequency (%)'},
    'zh': {'train':'训练集', 'validation':'验证集', 'test':'测试集',
           'mag':'震级 $M$', 'dis':r'震中距 ($\mathrm{km}$)',
           'frequency':r'样本占比 ($\%$)', 'count':'记录数量', 'azi':r'方位角 ($°$)',
           'rose_frequency':r'频率 ($\%$)'},
}


def setup(language):
    if language not in LABELS:
        raise ValueError('language must be zh or en')
    fonts = {}
    for name in ('SimSun','Times New Roman'):
        try:
            fonts[name] = fm.findfont(fm.FontProperties(family=name),fallback_to_default=False)
        except ValueError as exc:
            raise RuntimeError(f'Required font missing: {name}. Install this font; no substitution allowed.') from exc
    font = fm.FontProperties(fname=fonts['SimSun' if language=='zh' else 'Times New Roman'])
    latin = fm.FontProperties(fname=fonts['Times New Roman'])
    # Chinese labels are explicitly SimSun, math/units and tick labels TNR.
    for char in ''.join(LABELS[language].values()):
        if '\u4e00' <= char <= '\u9fff' and ord(char) not in FT2Font(fonts['SimSun']).get_charmap():
            raise RuntimeError(f'SimSun glyph missing: {char}')
    plt.rcParams.update({
        'font.family':'Times New Roman', 'font.size':8,
        'axes.labelsize':9, 'xtick.labelsize':8, 'ytick.labelsize':8,
        'legend.fontsize':8, 'axes.linewidth':.8, 'lines.linewidth':1.5,
        'xtick.major.width':.8, 'ytick.major.width':.8,
        'xtick.major.size':3, 'ytick.major.size':3,
        'figure.facecolor':'white', 'axes.facecolor':'white', 'savefig.facecolor':'white',
        'pdf.fonttype':42, 'ps.fonttype':42, 'pdf.use14corefonts':False,
        'mathtext.fontset':'custom', 'mathtext.rm':'Times New Roman',
        'mathtext.it':'Times New Roman:italic', 'mathtext.bf':'Times New Roman:bold',
        'mathtext.fallback':None,
        'axes.unicode_minus':True, 'hatch.linewidth':.35,
    })
    # FontProperties captures math_fontfamily at creation; explicitly update it
    # after rcParams so the first (Chinese) render cannot retain DejaVu defaults.
    font.set_math_fontfamily('custom')
    latin.set_math_fontfamily('custom')
    return font,latin,fonts


def save_figure(fig, folder, number, language):
    folder = Path(folder)
    suffix = 'CN' if language=='zh' else 'EN'
    stem = folder/f'Fig{number:02d}_{suffix}'
    # Explicit margins retain exact 190-mm canvas, unlike variable tight bbox.
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=600, facecolor='white')
    return [str(stem.with_suffix('.pdf')),str(stem.with_suffix('.png'))]
