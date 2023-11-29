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

# Function to resize image while maintaining aspect ratio and return an OffsetImage for plotting
def getImage(path, resize=True, max_length=40):
    if resize:
        # Open the image file
        img = Image.open(path)
        # Calculate the new size maintaining the aspect ratio
        ratio = float(max_length) / max(img.size)
        new_size = tuple([int(x*ratio) for x in img.size])
        # Resize the image
        img = img.resize(new_size, Image.ANTIALIAS)
        # Convert the image to array format for OffsetImage
        img = np.array(img)
    else:
        img = plt.imread(path)
    # Create an OffsetImage with the resized array
    return OffsetImage(img, zoom=1)  # 'zoom=1' because we already resized the image

# Your plotting code, where you call getImage(path) for each logo




df = nfl.import_pbp_data([2023], downcast=True, cache=False, alt_path=None)
df = create_explosive(df)
df = create_negative(df)
df = create_turnover(df)
df = create_distance(df)
df = create_downs(df)

'''
import os
import urllib.request


urls = pd.read_csv('https://raw.githubusercontent.com/statsbylopez/BlogPosts/master/nfl_teamlogos.csv')

for i in range(0,len(urls)):
    urllib.request.urlretrieve(urls['url'].iloc[i], os.getcwd() + FOLDER + urls['team_code'].iloc[i] + '.png')



logos = os.listdir(os.getcwd() + FOLDER)

logo_paths = []

for i in logos:
    logo_paths.append(os.getcwd() + FOLDER + str(i))

print(logo_paths)
'''
#df = import_clean_slice([2023], [1,2,3], 'SF', 'OFF', 'OF')

pd.set_option('display.max_columns', None)
#print(list(df.columns))

#df = pass_rate_by_personnel([2022], [1,2,3], 'SF', 'DEF', 'RZ')
#df = pass_rate_by_formation([2022], [1,2,3], 'SF', 'OF')
#df = pass_rate_by_personnel_and_formation([2023], [1,2,3], 'SF', 'OF')
#df = man_in_box_by_personnel([2023], [1,2,3], 'SF', 'DEF', 'OF')
#df = personnel_by_situation([2023], [1,2,3], 'SF', 'OFF', 'OF')
'''
############### 3rd down rates #######################
import pandas as pd

# Sample DataFrame
# df = pd.DataFrame({
#     'posteam': [...],
#     'down': [...],
#     'distance': [...]
# })

# Grouping and aggregating
import pandas as pd

# Step 1: Filter out rows where 'series' is NA
filtered_df = df.dropna(subset=['series'])

# Step 2: Group by team and game, then count unique series
series_count = filtered_df.groupby(['posteam', 'game_id'])['series'].nunique()

# Step 3: Summarize the counts across all games for each team
total_series_count = series_count.groupby('posteam').sum().rename('Total_Series_Count')

# Assuming df is your existing DataFrame

# First, filter the DataFrame for 'down' == 3
third_df = df[df['down'] == 3]
early_df = df[(df['down']==1)|(df['down']==2)]

total_third_downs = third_df.groupby('posteam').size().rename('Total_Third_Downs')


# Perform the individual count for each distance category
short_count = third_df[third_df['distance'] == 'Short'].groupby('posteam').size().rename('Short_Count')
medium_count = third_df[third_df['distance'] == 'Medium'].groupby('posteam').size().rename('Medium_Count')
long_count = third_df[third_df['distance'] == 'Long'].groupby('posteam').size().rename('Long_Count')
conversion_rate = third_df.groupby('posteam')['first_down'].mean().rename('Conversion_Rate')
early_down_convertion = early_df[early_df['first_down'] == 1].groupby('posteam').size().rename('Early_Down_Convertion')

# Combine the counts
third_counts_df = pd.concat([short_count, medium_count, long_count, total_third_downs,conversion_rate], axis=1).fillna(0)
early_counts_df = pd.concat([early_down_convertion, total_series_count], axis=1).fillna(0)

early_counts_df['Early_Down_Conversion_Rate'] = (early_counts_df['Early_Down_Convertion']) / (early_counts_df['Total_Series_Count'])

third_counts_df['Short_Percentage'] = (third_counts_df['Short_Count'] / third_counts_df['Total_Third_Downs']) * 100
third_counts_df['Medium_Percentage'] = (third_counts_df['Medium_Count'] / third_counts_df['Total_Third_Downs']) * 100
third_counts_df['Long_Percentage'] = (third_counts_df['Long_Count'] / third_counts_df['Total_Third_Downs']) * 100

# Reset the index to make 'posteam' a column
result_df = third_counts_df.reset_index()

result_df.drop(['Short_Count','Medium_Count','Long_Count','Total_Third_Downs'],axis=1, inplace=True)
early_counts_df.drop(['Early_Down_Convertion','Total_Series_Count'],axis=1, inplace=True)

df_sorted = result_df.sort_values(by=['Short_Percentage', 'Medium_Percentage', 'Long_Percentage'], ascending=[False, False, False])
early_sorted = early_counts_df.sort_values(by=['Early_Down_Conversion_Rate'],ascending=[False])

print(df_sorted)
print(early_sorted)

######################################################
'''

'''
############## Red zone plot #########################

redzone_drives = df[df['drive_inside20'] == 1]

result_df=redzone_drives[['posteam','defteam','play_id','drive_play_id_started','drive_play_id_ended','drive_inside20','fixed_drive_result','score_differential','score_differential_post']]
print(result_df)

redzone_visits = redzone_drives.groupby('posteam')['drive_play_id_ended'].nunique().reset_index(name='visits')

redzone_visits_sorted = redzone_visits.sort_values(by='visits',ascending=False)

#print(redzone_visits_sorted)

post_sum = redzone_drives[redzone_drives['drive_play_id_ended']==redzone_drives['play_id']].groupby('posteam')['score_differential_post'].sum()
pre_sum = redzone_drives[redzone_drives['drive_play_id_started']==redzone_drives['play_id']].groupby('posteam')['score_differential'].sum()

redzone_score = post_sum - pre_sum
redzone_score = redzone_score.sort_values(ascending=False).reset_index(name='points')

#print(redzone_score)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming redzone_visits_sorted and redzone_score are already defined as per your output
# We need to merge them on 'posteam' to align the visits with the corresponding points
merged_df = pd.merge(redzone_visits_sorted, redzone_score, on='posteam')

x = merged_df.visits
y = merged_df.points

# Now plot the scatter plot with a regression line (trend line)
fig, ax = plt.subplots(figsize=(15,15))
sns.regplot(x='visits', y='points', data=merged_df, fit_reg=True, ci=None, scatter_kws={'s': 0})

for posteam, visits, points in zip(merged_df['posteam'], merged_df['visits'], merged_df['points']):
    path = logo_paths.get(posteam)
    if path:
        imagebox = getImage(path)
        ab = AnnotationBbox(imagebox, (visits, points), frameon=False)
        ax.add_artist(ab)

# Set the title and labels of the plot
plt.title('Redzone Visits vs Points')
plt.xlabel('Redzone Visits')
plt.ylabel('Points Scored in Redzone')

# Show the plot
#plt.show()

merged_df['points_per_visit'] = merged_df['points']/merged_df['visits']
merged_df = merged_df.sort_values(by='points_per_visit',ascending=False)
print(merged_df)

#############################################################################
'''


#################### Turnovers takeaways plot ###############################
import pandas as pd

# Assuming df is your DataFrame and it has columns 'posteam', 'defteam', and 'turnover'

# First, calculate the turnovers by 'posteam'
turnovers = df.groupby('posteam')['turnover'].sum().reset_index(name='turnovers')
off_plays = df.groupby('posteam')['play'].sum().reset_index(name='off_plays') 
#TODO: play include penalties. Change to only run and pass
off_drives = df.groupby('posteam')['drive_play_id_ended'].nunique().reset_index(name='off_drives')

# Then, calculate the takeaways by 'defteam'
takeaways = df.groupby('defteam')['turnover'].sum().reset_index(name='takeaways')
def_plays = df.groupby('defteam')['play'].sum().reset_index(name='def_plays')
def_drives = df.groupby('defteam')['drive_play_id_ended'].nunique().reset_index(name='def_drives')

# Now, merge the two DataFrames on 'posteam' and 'defteam'
result_r = pd.merge(turnovers, off_plays, on='posteam', how='left')
result_off = pd.merge(result_r, off_drives, on='posteam', how='left')

result_l = pd.merge(takeaways, def_plays, on='defteam', how='left')
result_def = pd.merge(result_l, def_drives, on='defteam', how='left')

result = pd.merge(result_off, result_def, left_on='posteam', right_on='defteam', how='left')

# If you want to keep only the 'posteam' and 'takeaways' in the final result
# and rename 'posteam' to 'team' for clarity after the join
#result = result[['posteam', 'turnovers','takeaways']]

result['turnover_rate'] = result['turnovers'] / result['off_plays']
result['takeaway_rate'] = result['takeaways'] / result['def_plays']

result['turnover_perc'] = result['turnovers'] / result['off_drives']*100
result['takeaway_perc'] = result['takeaways'] / result['def_drives']*100

print(result)

import matplotlib.pyplot as plt
'''
### turnover
fig2, ax2 = plt.subplots(figsize=(15,15))
plt.scatter(result['takeaways'], result['turnovers'], alpha=0)
plt.gca().invert_yaxis()

negative_avg = result['takeaways'].mean()
explosive_avg = result['turnovers'].mean()

plt.axvline(x=negative_avg, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=explosive_avg, color='grey', linestyle='--', linewidth=1)
plt.xlabel('takeaways')
plt.ylabel('turnovers')
plt.title('Scatter Plot of turnovers vs. takeaways')

for posteam, takeaways, turnovers in zip(result['posteam'], result['takeaways'], result['turnovers']):
    path = logo_paths.get(posteam)
    if path:
        imagebox = getImage(path)
        ab = AnnotationBbox(imagebox, (takeaways, turnovers), frameon=False)
        ax2.add_artist(ab)

###turnover rate
fig3, ax3 = plt.subplots(figsize=(15,15))
plt.scatter(result['takeaway_rate'], result['turnover_rate'], alpha=0)
#plt.gca().invert_yaxis()

takeaways_rate_avg = result['takeaway_rate'].mean()
turnover_rate_avg = result['turnover_rate'].mean()

plt.axvline(x=takeaways_rate_avg, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=turnover_rate_avg, color='grey', linestyle='--', linewidth=1)
plt.xlabel('takeaway_rate')
plt.ylabel('turnover_rate')
plt.title('Scatter Plot of turnovers vs. takeaways')

for posteam, takeaway_rate, turnover_rate in zip(result['posteam'], result['takeaway_rate'], result['turnover_rate']):
    path = logo_paths.get(posteam)
    if path:
        imagebox = getImage(path)
        ab = AnnotationBbox(imagebox, (takeaway_rate, turnover_rate), frameon=False)
        ax3.add_artist(ab)
'''
### turnover perc
fig4, ax4 = plt.subplots(figsize=(15,15))
plt.scatter(result['takeaway_perc'], result['turnover_perc'], alpha=0)
#plt.gca().invert_yaxis()

takeaway_perc_avg = result['takeaway_perc'].mean()
turnover_perc_avg = result['turnover_perc'].mean()

plt.axvline(x=takeaway_perc_avg, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=turnover_perc_avg, color='grey', linestyle='--', linewidth=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_color('gray')
plt.gca().spines['left'].set_color('gray')
plt.tick_params(axis='both', colors='gray', which='both')
plt.xlabel('Takeaway Rate (%)', color='gray')
plt.ylabel('Turnover Rate (%)', loc='top', color='gray')

plt.plot(takeaway_perc_avg, turnover_perc_avg, 'o', color='gray', label='Ideal Point', alpha=1, markersize=5)
plt.annotate('League Average', (takeaway_perc_avg, turnover_perc_avg),
             textcoords="offset points", 
             xytext=(4,5), 
             ha='left',
             fontsize=10,
             color='gray')

plt.text(3.8, 20, 'Analyzing Team Performance in Turnovers and Takeaways in the NFL',
         horizontalalignment='left', verticalalignment='center', 
         fontsize=18, color='black', zorder=5)

plt.text(3.8, 19.5, 'How Contenders Stack Up in 2023 Through Week 12',
         horizontalalignment='left', verticalalignment='center', 
         fontsize=12, color='black', zorder=5)

plt.text(3.8, 4.6, 'Turnover Rate refers to the percentage of offensive drives that end in turnovers, indicating challenges in maintaining possession.',
         horizontalalignment='left', verticalalignment='center', 
         fontsize=10, color='black', zorder=5)
plt.text(3.8, 4.3, 'Takeaway Rate is the percentage of defensive drives resulting in takeaways, reflecting defensive success in regaining possession.',
         horizontalalignment='left', verticalalignment='center', 
         fontsize=10, color='black', zorder=5)

plt.text(20, 8, 'High Takeaway, Low Turnover\n(Better)', 
         horizontalalignment='right', verticalalignment='center', 
         fontsize=14, color='darkblue', zorder=5)

plt.text(7.75, 17.5, 'Low Takeaway, High Turnover\n(Worse)', 
         horizontalalignment='left', verticalalignment='center', 
         fontsize=14, color='red', zorder=5)
'''
plt.text(12.8, 6.5, 'High Takeaway, Low Turnover\n(Better)', 
         horizontalalignment='left', verticalalignment='center', 
         fontsize=14, color='blue', zorder=5)
'''
for posteam, takeaway_perc, turnover_perc in zip(result['posteam'], result['takeaway_perc'], result['turnover_perc']):
    path = logo_paths.get(posteam)
    if path:
        imagebox = getImage(path)
        ab = AnnotationBbox(imagebox, (takeaway_perc, turnover_perc), frameon=False)
        ax4.add_artist(ab)

plt.show()
###############################################


'''
########## Explosives-negatives plot ##########
import pandas as pd

result_df = df.groupby('posteam').agg({
    'explosive': 'sum',
    'negative': 'sum'
}).reset_index()

result_df = result_df.sort_values(by=['explosive', 'negative'], ascending=[False, True])

print(result_df)

def_df = df.groupby('defteam').agg({
    'explosive': 'sum',
    'negative': 'sum'
}).reset_index()

print(def_df)

import pandas as pd
import matplotlib.pyplot as plt

#teams_df=nfl.import_team_desc()              <<<<<<<<<<<<<

import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(15, 15))
plt.scatter(result_df['negative'], result_df['explosive'],alpha=0)

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


fig2,ax2 = plt.subplots(figsize=(15, 15))
plt.scatter(def_df['negative'], def_df['explosive'],alpha=0)

negative_avg = def_df['negative'].mean()
explosive_avg = def_df['explosive'].mean()

plt.axvline(x=negative_avg, color='grey', linestyle='--', linewidth=1)
plt.axhline(y=explosive_avg, color='grey', linestyle='--', linewidth=1)
plt.xlabel('Negatives')
plt.ylabel('Explosives')
plt.title('Defensive of Explosives vs. Negatives')

for defteam, negative, explosive in zip(def_df['defteam'], def_df['negative'], def_df['explosive']):
    path = logo_paths.get(defteam)
    if path:
        imagebox = getImage(path)
        ab = AnnotationBbox(imagebox, (negative, explosive), frameon=False)
        ax2.add_artist(ab)

plt.show()

#######################################################
'''


'''
########## Personnel usage plot ##########
df = df.drop('Play Count', axis=1)

df = df.drop(('E1st', 'Medium'))
df = df.drop(('4th'))

print(df)

import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your original DataFrame

# Convert the MultiIndex into a single string index
df['DownDistance'] = [' '.join(map(str, ind)) for ind in df.index]
df.reset_index(drop=True, inplace=True)
df.set_index('DownDistance', inplace=True)

# Define custom sort keys
down_order = {'P1st': 4, 'E1st': 3, '2nd': 2, '3rd': 1, '4th': 0}
distance_order = {'Long': 0, 'Medium': 1, 'Short': 2}

# Combine the custom sort keys into a single sort key
sort_key = pd.Series(df.index).apply(lambda x: (down_order[x.split()[0]], distance_order[x.split()[1]]))

# Sort the DataFrame by our custom sort key
df_sorted = df.iloc[sort_key.argsort()]

# Define shades of gray for each column
shades_of_gray = ['0.2', '0.4', '0.6', '0.8']

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

# The bottom parameter keeps track of where each bar starts (left edge)
bottom = pd.Series([0] * len(df_sorted), index=df_sorted.index)

# Iterate through the sorted DataFrame's columns to stack them
for i, column in enumerate(df_sorted.columns):
    ax.barh(df_sorted.index, df_sorted[column] * 100, left=bottom, label=column, color=shades_of_gray[i])
    bottom += df_sorted[column] * 100

# Formatting the plot
ax.set_xlabel('Percentage')
ax.set_ylabel('Down and Distance')
ax.set_xlim(0, 100)  # Since we're working with percentages, the x-axis should go from 0 to 100
ax.legend(title='Offense Personnel')

plt.show()

###############################################
'''
'''
df = import_clean_slice([2023], [1,2,3], 'SF', 'OFF', 'OF')

print(df.head())

pivot = pd.pivot_table(df, values=['pass','rush'], 
                                index=['Down', 'distance', 'offense_personnel', 'offense_formation'],
                                aggfunc={'pass': np.mean, 'rush': np.mean,'distance': len},
                                fill_value=0)

print(pivot)

  
pivot = pivot.divide(pivot.sum(axis=1), axis=0)

pivot = pivot.dropna(how='all')

pivot['Play Count'] = df.groupby(['Down', 'distance'])['play'].count()

print(pivot)

cols_over_10 = pivot.columns[pivot.gt(0.10).any()]

filtered_pivot = pivot[cols_over_10]

average_proportion = filtered_pivot.mean(axis=0)

sorted_columns = average_proportion.sort_values(ascending=False).index

sorted_pivot = filtered_pivot[sorted_columns]

print(sorted_pivot)

#pivot = order_pivot(pivot)
'''

'''
average_proportion = pivot.mean(axis=0)

# Step 2: Identify which personnel combinations have an average proportion > 5%
personnel_over_10 = average_proportion[average_proportion > 0.025].index

# Step 3: Filter the pivot table to only consider those personnel combinations
filtered_pivot = pivot[personnel_over_10]

print(filtered_pivot)
'''
'''pivot = np.round(pd.pivot_table(df_bal, values=['number_of_pass_rushers','defenders_in_box'], 
                                index=['Down', 'Distance', 'defense_personnel'], 
                                #columns=['Down'], 
                                aggfunc=np.mean,
                                fill_value=0),2)

print(pivot)'''

#Basic offensive personnel
'''pivot2 = np.round(pd.pivot_table(df_bal, values=['pass','rush'], 
                                index=['Down', 'Distance', 'offense_formation'], 
                                #columns=['down'], 
                                aggfunc=np.mean,
                                fill_value=0),2)'''

#print(pivot2)


'''passing = nfl.import_ngs_data('rushing', [2022])
#columns : required, type of data (passing, rushing, receiving)

pd.set_option('display.max_columns', None)

print(passing.head())'''

#print(nfl.see_weekly_cols())

'''win = nfl.import_sc_lines([2020])

print(win.head())'''

'''dc = nfl.import_depth_charts([2022])

print(dc.describe())'''

'''draft = nfl.import_draft_values()
print(draft.head(35))'''

'''weekly = nfl.import_weekly_data([2022])
print(weekly.columns.tolist())'''

'''
    Usage for game searching

    Previous matchups for instance
'''
'''
schedule = nfl.import_schedules(range(2005,2023))

#print(schedule.head())
#print(schedule.columns)

filtered_schedule = schedule[((schedule['home_team'] == 'KC') & (schedule['away_team'] == 'DET')) |
                             ((schedule['home_team'] == 'DET') & (schedule['away_team'] == 'KC'))]

# Sort by gameday and get the latest three games
result = filtered_schedule.sort_values(by='gameday', ascending=False).head(3)

columns_to_show = ['gameday', 'home_team', 'away_team', 'home_score', 'away_score']

# Filter the DataFrame for those columns
subset_result = result[columns_to_show]

print(subset_result)
'''

