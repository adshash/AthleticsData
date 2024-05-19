#%% Project Planning = Gantt Charts
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Dictionary of colour assignment
person_colours = {'Zoe': 'g',
                  'Katie': 'r',
                  'Reece': 'b',
                  'Adil': 'c',
                  'Adam': 'm',
                  'Ben': 'y',
                  'Kyle': '0.8'}

patches = []  # init patches

# adds the colour to the patch for each person
for person in person_colours:
    patches.append(mpatches.Patch(color=person_colours[person]))

# array of tasks
tasks = ['Gantt Chart',
         'Part B: Examine \nHistorical Trends',
         'Part B: Compare \n events against \ngender',
         'Part B: Linear \nRegression',
         'Part C: \nImplementation 1',
         'Part C: \nImplementation 2',
         'Part D: \nIntroduction',
         'Part D: Part B \n Write-up',
         'Part D: Part C \n Write-up',
         'Part D: \nConclusion',
         'Presentation \nSlideshow']


start_dates = ['2024-04-17',
               '2024-04-18',
               '2024-04-29',
               '2024-05-02',
               '2024-04-20',
               '2024-04-20',
               '2024-04-27',
               '2024-04-30',
               '2024-04-30',
               '2024-05-06',
               '2024-05-02']

end_dates = ['2024-04-18',
             '2024-05-02',
             '2024-05-04',
             '2024-05-05',
             '2024-05-04',
             '2024-05-04',
             '2024-04-30',
             '2024-05-09',
             '2024-05-06',
             '2024-05-09',
             '2024-05-09']

person = ['Adil', 'Adil', 'Ben', 'Adil', 'Adam', 'Reece', 'Zoe', 'Zoe', 'Katie', 'Katie', 'Adam']

# dataframe using the previous arrays as columns and prescribes corresponding labels
data = {'Tasks': tasks, 'Start Dates': start_dates, 'End Dates': end_dates, 'Person': person}
df = pd.DataFrame(data)

# converts start dates/end dates to dt objects
df['Start Dates'] = pd.to_datetime(df['Start Dates'])
df['End Dates'] = pd.to_datetime(df['End Dates'])


df = df.sort_values(by='Start Dates', ascending=True, ignore_index=True)

df['Task Duration'] = df['End Dates'] - df['Start Dates']

duration = df['End Dates'].max() - df['Start Dates'].min()  # Total duration of the project

fig, ax = plt.subplots()  # initialises a figure

ax.xaxis_date()  # registers x-axis as dates.

ax.barh(y=df['Tasks'], width=df['Task Duration'], left=df['Start Dates'])  # horizontal bar chart for task array

ax.set_title('Group Iota: Project Plan')
ax.set_xlabel('Date')
ax.set_ylabel('Task')
ax.set_xlim(df['Start Dates'].min(), df['End Dates'].max())  # sets the limits of the chart

fig.gca().invert_yaxis()  # Causes the 1st defined task to be at the top

ax.tick_params(axis='x', labelrotation=45)

#%% Gridlines for the x axis
ax.xaxis.grid(True, alpha=0.5)


#%% Adding a legend with the patches from the person colours dictionary
ax.legend(handles=patches, labels=person_colours.keys(), fontsize=11, bbox_to_anchor=(1, 0.5))


#%% adding a translucent bar for the whole task to allow for a solid overlay
for index, row in df.iterrows():
    ax.barh(y=row['Tasks'], width=row['Task Duration'], left=row['Start Dates'], color=
        person_colours[row['Person']], alpha=0.5)


#%% Adding completion bars
completion = [1, 1, 1, 1, 1, 1, 0.8, 1, 1, 0.5, 0.3]

df['Completed Days'] = completion*df['Task Duration']

for index, row in df.iterrows():
    ax.barh(y=row['Tasks'], width=row['Completed Days'], left=row['Start Dates'],
            color=person_colours[row['Person']])  # adds a completion bar

#%% Adding a blue line of current date
current_date = pd.to_datetime('2024-05-07')  # Custom date (using the datetime now method would cause the bar to be
# off-screen on the 9th.
ax.axvline(x=current_date, color='b')  # creates the line

ax.text(x=current_date + pd.Timedelta(days=-3.5), y=2, s=f"Current Date:{current_date.date()}", color='b')  # text for
# current date to include the current date on the line

# add completion percentages to tasks:
for n in range(len(completion)):
    ax.text(x=df['Start Dates'][n] + pd.Timedelta(days=1),
            y=ax.get_yticks()[n]+0.25,
            s=f"{completion[n]*100} % \ncomplete")
# for each completion bar, this adds text stating how completed each bar is.

plt.show()