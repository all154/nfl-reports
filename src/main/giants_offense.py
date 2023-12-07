import nfl_data_py as nfl
import pandas as pd
import numpy as np
from data_cleaner import *
from offense_tendencies import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

logo_paths = {
    'NE': '/home/all154/coding/nfl-reports/src/main/logos/NE.png',
    'MIA': '/home/all154/coding/nfl-reports/src/main/logos/MIA.png',
    'CLE': '/home/all154/coding/nfl-reports/src/main/logos/CLE.png',
    'TB': '/home/all154/coding/nfl-reports/src/main/logos/TB.png',
    'ATL': '/home/all154/coding/nfl-reports/src/main/logos/ATL.png',
    'CHI': '/home/all154/coding/nfl-reports/src/main/logos/CHI.png',
    'MIN': '/home/all154/coding/nfl-reports/src/main/logos/MIN.png',
    'NYJ': '/home/all154/coding/nfl-reports/src/main/logos/NYJ.png',
    'NO': '/home/all154/coding/nfl-reports/src/main/logos/NO.png',
    'DAL': '/home/all154/coding/nfl-reports/src/main/logos/DAL.png',
    'BAL': '/home/all154/coding/nfl-reports/src/main/logos/BAL.png',
    'LV': '/home/all154/coding/nfl-reports/src/main/logos/LV.png',
    'STL': '/home/all154/coding/nfl-reports/src/main/logos/STL.png',
    'IND': '/home/all154/coding/nfl-reports/src/main/logos/IND.png',
    'DEN': '/home/all154/coding/nfl-reports/src/main/logos/DEN.png',
    'CAR': '/home/all154/coding/nfl-reports/src/main/logos/CAR.png',
    'HOU': '/home/all154/coding/nfl-reports/src/main/logos/HOU.png',
    'KC': '/home/all154/coding/nfl-reports/src/main/logos/KC.png',
    'PIT': '/home/all154/coding/nfl-reports/src/main/logos/PIT.png',
    'GB': '/home/all154/coding/nfl-reports/src/main/logos/GB.png',
    'CIN': '/home/all154/coding/nfl-reports/src/main/logos/CIN.png',
    'BUF': '/home/all154/coding/nfl-reports/src/main/logos/BUF.png',
    'NYG': '/home/all154/coding/nfl-reports/src/main/logos/NYG.png',
    'WAS': '/home/all154/coding/nfl-reports/src/main/logos/WAS.png',
    'JAX': '/home/all154/coding/nfl-reports/src/main/logos/JAX.png',
    'TEN': '/home/all154/coding/nfl-reports/src/main/logos/TEN.png',
    'SF': '/home/all154/coding/nfl-reports/src/main/logos/SF.png',
    'ARI': '/home/all154/coding/nfl-reports/src/main/logos/ARI.png',
    'LAC': '/home/all154/coding/nfl-reports/src/main/logos/LAC.png',
    'LA': '/home/all154/coding/nfl-reports/src/main/logos/LA.png',
    'OAK': '/home/all154/coding/nfl-reports/src/main/logos/OAK.png',
    'SEA': '/home/all154/coding/nfl-reports/src/main/logos/SEA.png',
    'DET': '/home/all154/coding/nfl-reports/src/main/logos/DET.png',
    'SD': '/home/all154/coding/nfl-reports/src/main/logos/SD.png',
    'PHI': '/home/all154/coding/nfl-reports/src/main/logos/PHI.png'
}


from PIL import Image
from matplotlib.offsetbox import OffsetImage

def getImage(path, resize=True, max_length=40):
    if resize:
        img = Image.open(path)
        ratio = float(max_length) / max(img.size)
        new_size = tuple([int(x*ratio) for x in img.size])
        img = img.resize(new_size, Image.ANTIALIAS)
        img = np.array(img)
    else:
        img = plt.imread(path)
    return OffsetImage(img, zoom=1)  


df = nfl.import_pbp_data(list(range(2004, 2024)), downcast=True, cache=False, alt_path=None)
df = df[df['week'] <= 12]
df = create_explosive(df)
df = create_negative(df)

import pandas as pd

result_df = df.groupby(['posteam', 'season']).agg({
    'explosive': 'sum',
    'negative': 'sum'
}).reset_index()

result_df = result_df.sort_values(by=['explosive', 'negative'], ascending=[False, True])
mask = ((result_df['posteam']=='NYG')&(result_df['season']==2023))
filtered_df = result_df[~mask]

plot_df = filtered_df.groupby(['explosive', 'negative']).size().reset_index(name='count')

plot_df = plot_df.sort_values(by='count')

import pandas as pd
import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(15, 15))
plt.scatter(result_df['negative'], result_df['explosive'],alpha=0)
plt.scatter(plot_df['negative'], plot_df['explosive'],alpha=1, s=30*plot_df['count'], color='blue')


negative_avg = result_df['negative'].mean()
explosive_avg = result_df['explosive'].mean()

plt.axvline(x=negative_avg, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=explosive_avg, color='grey', linestyle='--', linewidth=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_color('gray')
plt.gca().spines['left'].set_color('gray')
plt.tick_params(axis='both', colors='gray', which='both')
plt.xlabel('Negatives', color='gray')
plt.ylabel('Explosives', loc='top', color='gray')

y_label_ax_pos = ax.get_yaxis().get_label().get_position()

bbox = ax.get_window_extent().transformed(plt.gcf().transFigure.inverted())
y_label_fig_x = bbox.x0 + (y_label_ax_pos[0] * bbox.width)
y_label_fig_y = bbox.y0 + (y_label_ax_pos[1] * bbox.height)

horizontal_offset = -0.038
vertical_offset_for_title = 0.035
vertical_offset_for_subtitle = 0.015

# Title
plt.text(y_label_fig_x + horizontal_offset, y_label_fig_y + vertical_offset_for_title, 
         'Historical Struggles of the 2023 Giants Offense',
         horizontalalignment='left', fontsize=18, color='black', zorder=5, 
         transform=plt.gcf().transFigure)

# Subtitle
plt.text(y_label_fig_x + horizontal_offset, y_label_fig_y + vertical_offset_for_subtitle, 
         'Alone in Adversity - Charting the Worst Performance in Two Decades Up to Week 12',
         horizontalalignment='left', verticalalignment='center', 
         fontsize=12, color='black', zorder=5, 
         transform=plt.gcf().transFigure)

bbox = ax.get_window_extent().transformed(plt.gcf().transFigure.inverted())
footnote_start_x = bbox.x0 
footnote_start_y = bbox.y0 

horizontal_offset = -0.02  
vertical_offset = -0.06
vertical_offset_per_line = 0.02  

footnote_texts = [
    'Explosives: This category represents offensive plays resulting in significant yardage gains, specifically runs of more than 10 yards and passes exceeding 15 yards.',
    'Negatives: This metric tracks offensive plays that resulted in a loss of yards, including both sacks and tackles for loss (TFLs).',
    'Data Source: Data for this plot has been compiled from official NFL play-by-play records, covering the last 20 seasons up to week 12, inclusive of the 2023 season.'
]

for i, text in enumerate(footnote_texts):
    plt.text(footnote_start_x + horizontal_offset, 
             footnote_start_y + vertical_offset - i * vertical_offset_per_line, text,
             horizontalalignment='left', verticalalignment='center', 
             fontsize=10, color='gray', zorder=5, 
             transform=plt.gcf().transFigure)

negative = result_df['negative'][mask]
explosive = result_df['explosive'][mask]

imagebox = getImage(logo_paths.get('NYG'))
ab = AnnotationBbox(imagebox, (negative, explosive), frameon=False)
ax.add_artist(ab)

plt.show()

#######################################################