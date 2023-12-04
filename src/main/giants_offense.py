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


df = nfl.import_pbp_data(list(range(2022, 2024)), downcast=True, cache=False, alt_path=None)
df = df[df['week'] <= 12]
df = create_explosive(df)
df = create_negative(df)
#df = create_turnover(df)
#df = create_distance(df)
#df = create_downs(df)

########## Explosives-negatives plot ##########
import pandas as pd

result_df = df.groupby(['posteam', 'season']).agg({
    'explosive': 'sum',
    'negative': 'sum'
}).reset_index()

result_df = result_df.sort_values(by=['explosive', 'negative'], ascending=[False, True])

print(result_df)

import pandas as pd
import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(15, 15))
plt.scatter(result_df['negative'], result_df['explosive'],alpha=1)

negative_avg = result_df['negative'].mean()
explosive_avg = result_df['explosive'].mean()

plt.axvline(x=negative_avg, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=explosive_avg, color='grey', linestyle='--', linewidth=1)
plt.xlabel('Negatives')
plt.ylabel('Explosives')
plt.title('Scatter Plot of Explosives vs. Negatives')

for posteam, negative, explosive in zip(result_df['posteam'], result_df['negative'], result_df['explosive']):
    path = logo_paths.get(posteam)
    if path:
        imagebox = getImage(path)
        ab = AnnotationBbox(imagebox, (negative, explosive), frameon=False)
        ax.add_artist(ab)

plt.show()

#######################################################
