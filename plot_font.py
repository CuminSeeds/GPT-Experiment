"""保留 Matplotlib 默认绘图风格，仅统一文字为宋体、黑色。"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager


def apply_report_font():
    path = Path('C:/Windows/Fonts/simsun.ttc')
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        family = 'SimSun'
    plt.rcParams.update({
        'font.family': [family],
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })
