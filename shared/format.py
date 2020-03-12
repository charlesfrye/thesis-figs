from cycler import cycler
import matplotlib as mpl

LINEWIDTH = 4
FONTSIZE = 16

cal_colors = ['#003262', '#FDB515']
default_colors = [color["color"] for color in mpl.rcParamsDefault["axes.prop_cycle"][2:]]

mpl.rcParams['axes.prop_cycle'] = cycler(color=cal_colors + default_colors)