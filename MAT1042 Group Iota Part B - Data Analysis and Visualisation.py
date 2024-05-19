import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics

#%% Path definition for the sheets

pathAthletics = "Athletics data 2024.xlsx"


#%% Function's definition
def TimeToSeconds(timeValue):  # Converts hh:mm:ss times into seconds
    return timeValue.hour * 3600 + timeValue.minute * 60 + timeValue.second


def getUnits(valueColumn):
    UnitsArray = ""
    found = False
    for char in valueColumn:
        if found:
            UnitsArray += char
        if char == "(":
            found = True
    UnitsArray = UnitsArray[:-1]
    return UnitsArray


def distanceToFloat(distance):
    return float(distance)
def DateToYears(date):
    return date.year


def floatYearToDatetime(yearFloat):
    yearInt = int(yearFloat)  # Convert float to integer
    yearDatetime = dt.datetime(year=yearInt, month=6, day=1)  # Create dt object with the year half way through the year to minimise errors
    return yearDatetime.year

def StrToYears(date):  # For dfW100, the date given is not a data time
    YearValue = date[-4:-1] + date[-1]  # indexing the year from each string
    DateFormat = '%Y'  # default year formating
    YearValue = dt.datetime.strptime(YearValue, DateFormat)  #  creates dt object
    return YearValue.year


def StrToYearsWLJ(date):  # For dfWlj, the date given is not a datetime obj, returns the Year as a dt obj
    YearValue = date[-7:-3]
    DateFormat = '%Y'
    YearValue = dt.datetime.strptime(YearValue, DateFormat)
    return YearValue.year


def ColumnsDropper(dataframe, Columns):  # Drops a given array of columns and removes empty cells & Returns the df
    dataframe.drop(inplace=True, columns=Columns)  # Drops the defined array of columns
    dataframe.dropna(inplace=True)  # Removes any empty cells in the df


def LongestTimeBetweenRecord(df, eventName, valueColumn, positive):  # Finds how quickly a new record is broken in a given
    # data frame. The positive input is to differentiate between df's where positive diff = improvement or not.
    df['ValueDiff'] = df[valueColumn].diff()  # creates a column of differences from one cell to the previous
    endDate = None
    startDate = None
    NumbOfConsecutiveYears = 1
    MaxConsecutive = 0
    if positive == True:  # This uses the positive input to allow for the next iteration to apply to either type of diff
        df['ValueDiff'] *= -1  # changes the row to be the negative - allowing it to be treated
        # as an event where negative diff is improvement
    for index, row in df.iterrows():  # reiterates through each row in a df
        if row['ValueDiff'] >= 0 and df.loc[index+1, 'ValueDiff'] < 0:
            # this condition is met given that a row has no improvement AND the row after HAS an improvement
            # this makes this current row the end of the period of no improvement
            NumbOfConsecutiveYears += 1  # This increments how many years so the end of the period is included
            if NumbOfConsecutiveYears > MaxConsecutive:
                # This condition is met when the number of consecutive years is the NEW longest length without improvement
                MaxConsecutive = NumbOfConsecutiveYears  # This stores the new longest time
                endDate = row['Year']  # This stores the current date as the end of such improvement
                startDate = df.loc[index-NumbOfConsecutiveYears+1, 'Year']  # This causes the start date to be
                # the exact the period started rather than the day time frame before.
        elif row['ValueDiff'] >= 0 and df.loc[index+1, 'ValueDiff'] >= 0:
            # This condition is met when there is a period of no improvement and the following cell also has no improve
            NumbOfConsecutiveYears += 1
        else:
            NumbOfConsecutiveYears = 1
            # This condition is met when the difference is py.NaN which occurs for the first cell since the diff()
            # method subtracts from the cell before (there is no cell before the first one)
    print(f"The longest time between records for {eventName} was {MaxConsecutive} Years. Starting from {startDate} and finishing {int(endDate)}")
    return


def ShortestTimeBetweenRecord(df, valueColumn, positive):  # This function finds the shortest time before a record is broken
    df['ValueDiff'] = df[valueColumn].diff()  # creates a column for the differences between a cell and the cell before
    endDate = None
    startDate = None
    numOfYears = 1
    yearsUntilRecord = np.inf  # init - infinity so that the first record break counts as a record break and if == true
    if positive == False:  # This uses the positive input to allow for the next iteration to apply to either type of diff
        df['ValueDiff'] *= -1  # changes the row to be the negative - allowing it to be treated
        # as an event where negative diff is improvement
    print(df)
    for index, row in df.iterrows():
        if row['ValueDiff'] >= 0:  # if theres improvement,the numOfYears increases by 1
            numOfYears += 1
        if row['ValueDiff'] < 0:  # if there is an improvement
            if numOfYears < yearsUntilRecord:
                # if the improvement occurred faster than previous shortest
                yearsUntilRecord = numOfYears  # this becomes new shortest
                endDate = row['Year']  # the end date is this row
                startDate = df.loc[index - numOfYears, 'Year']  # start date is numOfYears before
            numOfYears = 1  # returns num of years to 1 to continue the iteration
    print(yearsUntilRecord, endDate, startDate)
    return  # return isn't working on my pc for some reason so im using print


#%% Plotting of DataFrames
def Plot(df, eventName, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df[ylabel], marker='o', linestyle='-')
    plt.title(f"{eventName} Progression Over the Years")
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(df['Year'], rotation=45)
    years = df['Year'].tolist()
    plt.xticks(range(min(years), max(years) + 1, 4))  # Assuming data spans multiple years
    plt.tight_layout()
    plt.show()


#%% Average rate of improvement and largest differences functions
def avgRateOfImprovement(df, eventName, valueCol, positive):  # Finds the largest length of time between improvements

    df.sort_values(by='Year')  # sorts the df by year (arbitrary since data is already sorted by year but if it ever
    # wasn't, it would be now)

    df['value_diff'] = df[valueCol].diff()  # this creates a column for the dfferences between a cell and cell before
    df['year_diff'] = df['Year'].diff()  # " "  " but for years
    if positive == False:  # This uses the positive input to allow for the next iteration to apply to either type of diff
        df['value_diff'] *= -1  # changes the row to be the negative - allowing it to be treated
    df.loc[df['year_diff'] == 0, 'year_diff'] = np.nan  # Removes any 0 values to remove infinity error
    df.loc[df['value_diff'] == 0, 'value_diff'] = np.nan
    for index, row in df.iterrows():
        if index != 0:
            previousValue = float(df.loc[index-1, valueCol])
            df.loc[index, 'Proportional Improvement'] = row['value_diff'] / previousValue
        else:
            row['Proportional Improvement'] = np.nan
    df['improvementRate'] = df['Proportional Improvement'] / df['year_diff']  # Calculates rates of improvement in each

    df.loc[df['year_diff'] == 0, 'improvementRate'] = np.nan  # This also reduces infinity errors

    averageImprovRate = df['improvementRate'].mean()  # Calculates mean of improvement
    percentageRate = averageImprovRate * 100
    percentageRate = round(percentageRate, 4)
    print(f'The average percentage rate improvement for {eventName} was {percentageRate}% per year')
    return  # This doesn't return values for some reason but its fine bc of print function :)


def HighestJumpInValue(df, yearCol, valueCol):

    df.sort_values(by=yearCol)  # Once again sorts the df

    df['valueDiff'] = df[valueCol].diff()  # Creates a differences column for the value

    indexOfMax = df['valueDiff'].idxmax()  # Find the index of the maximum difference

    print("The year with the largest improvement was", df.loc[indexOfMax, yearCol])  # Prints year for max difference

    return


def LowestJumpInValue(df, yearCol, valueCol):

    df.sort_values(by=yearCol)  # Once again sorts the df

    df['valueDiff'] = df[valueCol].diff()  # Creates a differences column for the value
    df['valueDiff'] = df['valueDiff'].abs()
    indexOfMin = df['valueDiff'].idxmin()  # Find the index of the minimum difference

    print("The year with the smallest improvement was from", df.loc[indexOfMin - 1, yearCol],"to", df.loc[indexOfMin, yearCol])  # Prints year for min difference

    return


#%% The splitting of mens100 Sheet into mens100 and olympics
dfM100 = pd.read_excel(pathAthletics, sheet_name="men's 100m", header=1)  # df for mens 100 M
ColumnsToDropOlympic = ["winner", "Athlete", "Date",
                        "Time (seconds)", "Unnamed: 6",
                        "height (metres)", "weight (kg)", "BMI (Body Mass Index)"]  # Cleans the data/splits it
dfOlympic = dfM100.drop(columns=ColumnsToDropOlympic)  # Separates out the table on sheet 1
ColumnsToDropMens100 = ["year", "winner", "time",
                        "height (metres)", "weight (kg)",
                        "BMI (Body Mass Index)", "Unnamed: 6", "Athlete"]  # Columns that require cleaning
ColumnsDropper(dfM100, ColumnsToDropMens100)  # Drops the columns or dfM100



#%% The splitting of Marathon for mens into WR and fastest each year

MMaraWR = pd.read_excel(pathAthletics,
                        sheet_name="mens marathon", header=1)  # mens marathon WR sheet acts as a spine for both
ColumnsToDropWRMarathon = ["year", "country", "athlete",
                           "athlete.1", "country.1", "time",
                           "Unnamed: 4", "Unnamed: 5"]  # These are the columns that are dropped to make the WR times
ColumnsToDropFastMarathon = ['date', 'seconds', 'athlete',
                             'country', "Unnamed: 4",
                             "Unnamed: 5", 'athlete.1',
                             'country.1']  # These are the columns that are dropped to make the Fastest times df
MMaraFast = MMaraWR.drop(columns=ColumnsToDropFastMarathon)  # This creates the Fastest Times df
ColumnsDropper(MMaraWR, ColumnsToDropWRMarathon)

#%% The splitting of Marathon for womens into WR and fastest each yeah

WMaraWR = pd.read_excel(pathAthletics,
                        sheet_name="women's marathon", header=3)  # marathon WR sheet acts as a spine for both
ColumnsToDropWRMarathon = ["Unnamed: 0", "Unnamed: 4", "Unnamed: 5", "Year", "Athlete            ",
                           "Race Time - Seconds", "Date ",
                           "Athlete"]  # These are the columns that are dropped to make the WR times
ColumnsToDropFastMarathon = ['Year.1', 'Time \n(Hours:Minutes:Seconds)', 'Athlete',
                             'Date ', 'Unnamed: 0', 'Unnamed: 4',
                             'Unnamed: 5',
                             'Athlete            ']  # These are the columns that are dropped to make the Fastest times df
WMaraFast = WMaraWR.drop(columns=ColumnsToDropFastMarathon)  # This creates the Fastest Times df
WMaraWR.drop(inplace=True, columns=ColumnsToDropWRMarathon)  # This finalises the WR Times df
WMaraWR.dropna(inplace=True)  # This cleans up the WR Times df

#%% Formatting of each database created so far

WMaraFast.rename(inplace=True,
                 columns={"Race Time - Seconds": "Time (seconds)"})  # Renames the Time column

WMaraWR.rename(inplace=True,
               columns={"Year.1": "Year", "Time \n(Hours:Minutes:Seconds)": "Time (seconds)"})  # Renames column

WMaraWR['Year'] = WMaraWR['Year'].apply(floatYearToDatetime)  # Turns the year column into datetime objects
WMaraWR['Time (seconds)'] = WMaraWR['Time (seconds)'].apply(TimeToSeconds)  # Turns the time column into seconds

WMaraWR.loc[32, 'Year'] = 2017  # Rewrites the mistake in the df to 2017 -
# https://results.london-marathon.co.uk/2017/?pid=list

MMaraFast['time'] = MMaraFast['time'].apply(TimeToSeconds)  # Turns the time column to seconds
MMaraFast.rename(inplace=True, columns={"time": "Time (seconds)", "year": "Year"})  # Renames the time column

MMaraWR['date'] = MMaraWR['date'].apply(DateToYears)
MMaraWR.rename(inplace=True, columns={'date': 'Year', 'seconds': 'Time (seconds)'})  # Changes Column name


dfM100['Date'] = dfM100['Date'].apply(DateToYears)  # Changes all the dates into years for M100
dfM100.rename(inplace=True,columns={'Date': 'Year', 'seconds': 'Time (seconds)'})  # Changes column names

dfOlympic.rename(inplace=True, columns={'year': 'Year', 'time': 'Time (seconds)'})  # Renames the columns


#%% Creation of databases

# Women's 100m df
dfW100 = pd.read_excel(pathAthletics, sheet_name="womens 100m", header=1)

columnsToDropW100 = ['Athlete', 'Nationality', 'Location']  # Creates an array of columns that are unnecessary
ColumnsDropper(dfW100, columnsToDropW100)  # Drops the array of columns

dfW100['Date'] = dfW100['Date'].apply(StrToYears)  # Changes the dates column into years
dfW100.rename(inplace=True, columns={'Date': 'Year', 'Time': 'Time (seconds)'})  # Renames the columns to 'Year'
dfW100.sort_values(by='Year', ascending=True, inplace=True)  # Due to the unsorted Date column

# Men's 1500m df
dfM1500 = pd.read_excel(pathAthletics, sheet_name="men's 1500m",header=1)  # Creates df from the excel sheet

columnsToDropM1500 = ['time', 'athlete']  # array of columns to drop
ColumnsDropper(dfM1500, columnsToDropM1500)  # drops the columns from the array

dfM1500['date'] = dfM1500['date'].apply(DateToYears)  # applies the datetoyears function to format the year
dfM1500.rename(inplace=True, columns={'date': 'Year', 'seconds': 'Time (seconds)'})  # renames columns


# Women's 1500m df
dfW1500 = pd.read_excel(pathAthletics, sheet_name="women's 1500m", header=2)  # Creates df from sheet

columnsToDropW1500 = ['place', 'Unnamed: 3', 'athlete']  # array of columns to drop
ColumnsDropper(dfW1500, columnsToDropW1500)  # drops the columns from the array

dfW1500['date'] = dfW1500['date'].apply(DateToYears)  # changes the dates in the column to years

dfW1500['time'] = dfW1500['time'].apply(TimeToSeconds)
dfW1500.rename(inplace=True, columns={'date': 'Year', 'time': 'Time (seconds)'})  # renames the column to year


# Men's Long Jump df
dfMlj = pd.read_excel(pathAthletics, sheet_name="men's long jump", header=1)  # Creates df from sheet

columnsToDropMlj = ['name', 'country', 'Unnamed: 3']  # Array of columns to drop
ColumnsDropper(dfMlj, columnsToDropMlj)  # Drops array of columns

dfMlj.rename(inplace=True, columns={'year': 'Year', 'distance': 'Distance (m)'})  # Renames the columns


# Women's Long Jump df
dfWlj = pd.read_excel(pathAthletics, sheet_name="women's long jump", header=2)  # Creates df of Women's lj

columnsToDropWlj = ['Athlete', 'Venue']  # Array of columns to drop
ColumnsDropper(dfWlj, columnsToDropWlj)  # Drops array of columns

dfWlj['Date'] = dfWlj['Date'].apply(StrToYearsWLJ)  # Changes the date column to years and corrects the type

dfWlj.rename(inplace=True, columns={'Date': 'Year', 'distance (m)': 'Distance (m)'})  # Renames the columns

dfWlj['Distance (m)'] = dfWlj['Distance (m)'].apply(distanceToFloat)


# Men's High Jump df
dfMhj = pd.read_excel(pathAthletics, sheet_name="men's high jump", header=1)   # Creates df of Men's hj

columnsToDropMhj = ['name', 'Unnamed: 3', 'country']  # Array of columns to drop
ColumnsDropper(dfMhj, columnsToDropMhj)  # Drops array of columns

dfMhj.rename(inplace=True, columns={'year': 'Year', 'distance': 'Height (m)'})

# Women's High Jump df
dfWhj = pd.read_excel(pathAthletics, sheet_name="women's high jump", header=1)  # Creates df of women's hj

columnsToDropWhj = ['Athlete', 'Place', 'Unnamed: 0']  # Array of columns to drop
ColumnsDropper(dfWhj, columnsToDropWhj)  # Drops array of columns

dfWhj['Date'] = dfWhj['Date'].apply(DateToYears)  # Changes the date column to years and corrects the type

dfWhj.rename(inplace=True, columns={'Date': 'Year'})



#%% Comparing Men's and women's progression
list_of_times = dfW100['Time (seconds)'].tolist()  # converting 100m womentimes into a list
length_of_list = len(list_of_times)  # setting variable equal to size of list
i = 1
new_array = [0]
first_record = list_of_times[0]
while i != length_of_list:
    percentage = (1 - (
                list_of_times[i] / first_record)) * 100  # working out percentage difference for each record compared to the first record
    new_array.append(percentage)  # adding this percent to an array
    i = i + 1
percentchance = np.array(new_array)
dfW100['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

list_of_times2 = dfM100['Time (seconds)'].tolist()  # converting 100m men times into a list
length_of_list2 = len(list_of_times2)  # setting variable equal to size of list
j = 1
new_array2 = [0]
first_record2 = list_of_times2[0]
while j != length_of_list2:
    percentag = (1 - (
                list_of_times2[j] / first_record2)) * 100  # working out percentage difference for each record compared to the first record
    new_array2.append(percentag)  # adding this percent to an array
    j = j + 1
percentchance = np.array(new_array2)
dfM100['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

# This will create the columns for the men and women 100m dataframes

list_of_times = dfM1500['Time (seconds)'].tolist()  # converting 1500m men times into a list
length_of_list = len(list_of_times)
i = 1
new_array = [0]
first_record = list_of_times[0]
while i != length_of_list:
    percentage = (1 - (
                list_of_times[i] / first_record)) * 100  # working out percentage difference for each record compared to the first record
    new_array.append(percentage)
    i = i + 1
percentchance = np.array(new_array)  # adding this percent to an array
dfM1500['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

list_of_times2 = dfW1500['Time (seconds)'].tolist()  # converting 1500m women times into a list
length_of_list2 = len(list_of_times2)
j = 1
new_array2 = [0]
first_record2 = list_of_times2[0]
while j != length_of_list2:
    percentag = (1 - (
                list_of_times2[j] / first_record2)) * 100  # working out percentage difference for each record compared to the first record
    new_array2.append(percentag)  # adding this percent to an array
    j = j + 1
percentchance = np.array(new_array2)
dfW1500['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

# This will create the columns for the men and women long jump dataframes

list_of_times = dfMlj['Distance (m)'].tolist()  # converting men long jump distances into a list
length_of_list = len(list_of_times)
i = 1
new_array = [0]
first_record = list_of_times[0]
while i != length_of_list:
    percentage = ((list_of_times[
                       i] / first_record) - 1) * 100  # working out percentage difference for each record compared to the first record
    new_array.append(percentage)  # adding this percent to an array
    i = i + 1
percentchance = np.array(new_array)
dfMlj['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

list_of_times2 = dfWlj['Distance (m)'].tolist()  # converting women long jump distances into a list
length_of_list2 = len(list_of_times2)
j = 1
new_array2 = [0]
first_record2 = list_of_times2[0]
while j != length_of_list2:
    percentag = (float(list_of_times2[j]) / float(
        first_record2) - 1) * 100  # working out percentage difference for each record compared to the first record
    new_array2.append(percentag)  # adding this percent to an array
    j = j + 1
percentchance = np.array(new_array2)
dfWlj['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

# This will create the columns for the men and women high jump dataframes

list_of_times = dfMhj['Height (m)'].tolist()  # converting men high jump heights into a list
length_of_list = len(list_of_times)
i = 1
new_array = [0]
first_record = list_of_times[0]
while i != length_of_list:
    percentage = ((list_of_times[
                       i] / first_record) - 1) * 100  # working out percentage difference for each record compared to the first record
    new_array.append(percentage)  # adding this percent to an array
    i = i + 1
percentchance = np.array(new_array)
dfMhj['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

list_of_times2 = dfWhj['Height (m)'].tolist()  # converting women high jump heights into a list
length_of_list2 = len(list_of_times2)
j = 1
new_array2 = [0]
first_record2 = list_of_times2[0]
while j != length_of_list2:
    percentag = (float(list_of_times2[j]) / float(
        first_record2) - 1) * 100  # working out percentage difference for each record compared to the first record
    new_array2.append(percentag)  # adding this percent to an array
    j = j + 1
percentchance = np.array(new_array2)
dfWhj['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

# This will create the columns for the men and women marathon records dataframes

list_of_times = MMaraWR['Time (seconds)'].tolist()  # converting 1500m men times into a list
length_of_list = len(list_of_times)
i = 1
new_array = [0]
first_record = list_of_times[0]
while i != length_of_list:
    percentage = (1 - (
                list_of_times[i] / first_record)) * 100  # working out percentage difference for each record compared to the first record
    new_array.append(percentage)
    i = i + 1
percentchance = np.array(new_array)  # adding this percent to an array
MMaraWR['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes

list_of_times2 = WMaraWR['Time (seconds)'].tolist()  # converting 1500m women times into a list
length_of_list2 = len(list_of_times2)
j = 1
new_array2 = [0]
first_record2 = list_of_times2[0]
while j != length_of_list2:
    percentag = (1 - (
                list_of_times2[j] / first_record2)) * 100  # working out percentage difference for each record compared to the first record
    new_array2.append(percentag)  # adding this percent to an array
    j = j + 1
percentchance = np.array(new_array2)
WMaraWR['percentage'] = percentchance.tolist()  # adding a new column with the percentage changes tp the dataframes


def ComparePlot(dfmen, dfwomen, jointeventName, ylabel):
    plt.figure(figsize=(10, 6))  # Adjust size if needed
    plt.plot(dfmen['Year'], dfmen[ylabel], marker='o', linestyle='-', label= "Men")
    plt.plot(dfwomen['Year'], dfwomen[ylabel], marker='o', linestyle='-', label = "Women")
    plt.title(f"{jointeventName}")
    plt.xlabel('Year')
    plt.ylabel("Percentage change/improvement of records")
    plt.grid(True)
    plt.xticks(dfmen['Year'], rotation=45)  # Adjust rotation if needed
    min_year =  dfmen['Year'].tolist()
    max_year = WMaraWR['Year'].tolist()
    plt.xticks(range(min(min_year), max(max_year)+1, 4))  # Assuming data spans multiple years
    plt.tight_layout()  # Adjust layout
    plt.legend(loc='center right')
    plt.show()


def CompareMenandWomenAllEventsPlot():
    plt.figure(figsize=(10, 9))  # Adjust size if needed
    plt.plot(MMaraWR['Year'], MMaraWR['percentage'], marker='o', linestyle='-', label="Men's marathon")
    plt.plot(WMaraWR['Year'], WMaraWR['percentage'], marker='o', linestyle='-', label="Women's marathon")
    plt.plot(dfM1500['Year'], dfM1500['percentage'], marker='o', linestyle='-', label="Men's 1500m")
    plt.plot(dfW1500['Year'], dfW1500['percentage'], marker='o', linestyle='-', label="Women's 1500m")
    plt.plot(dfMlj['Year'], dfMlj['percentage'], marker='o', linestyle='-', label="Men's long jump")
    plt.plot(dfWlj['Year'], dfWlj['percentage'], marker='o', linestyle='-', label="Women's long jump")
    plt.plot(dfMhj['Year'], dfMhj['percentage'], marker='o', linestyle='-', label="Men's high jump")
    plt.plot(dfWhj['Year'], dfWhj['percentage'], marker='o', linestyle='-', label="Women's high jump")
    plt.plot(dfM100['Year'], dfM100['percentage'], marker='o', linestyle='-', label="Men's 100m")
    plt.plot(dfW100['Year'], dfW100['percentage'], marker='o', linestyle='-', label="Women's 100m")
    plt.title("Men's and Women's all events record Progression Over the Years")
    plt.xlabel('Year')
    plt.ylabel("Percentage change/improvement of records")
    plt.grid(True)
    plt.xticks(dfM100['Year'], rotation=45)  # Adjust rotation if needed
    min_year = dfMlj['Year'].tolist()
    max_year = WMaraWR['Year'].tolist()
    plt.xticks(range(min(min_year), max(max_year) + 1, 4))  # Assuming data spans multiple years
    plt.tight_layout()  # Adjust layout
    plt.legend(loc='center right')
    plt.show()


#%% Linear Regression
def linearRegression(df, eventName, valueColumn):
    x1data = np.array(df['Year'])  # Creates an array of the years
    yData = np.array(df[valueColumn])  # Creates a vector for the values
    linReg = LinearRegression(fit_intercept=True)  # scikit function to include Beta_0
    x1DataReshape = x1data.reshape(-1, 1)  # reshapes the data into a column array
    linReg.fit(x1DataReshape, yData)  # fits the data
    yPredictedSk = linReg.predict(x1DataReshape)  # creates predicted values
    beta_0 = linReg.intercept_
    beta_1 = linReg.coef_
    label = f"(Fitted) Linear Regression Line \ny = {np.round(beta_0, 2)} + {np.round(beta_1[0], 2)}x"  # Label to include the function
    plt.scatter(x1DataReshape, yData, label="Observed Value")
    plt.plot(x1DataReshape, yPredictedSk, 'Orange', label=label)
    plt.xlabel('Year')
    plt.ylabel(valueColumn)
    plt.title(f"Regression for {eventName}")
    plt.vlines(x1data, yPredictedSk, yData, color='k',label="Error")
    plt.legend()
    plt.show()


#%% Regression Lines
linearRegression(dfWhj, "Women's High Jump", "Height (m)")
linearRegression(dfWlj, "Women's Long Jump", "Distance (m)")
linearRegression(dfMhj, "Men's High Jump", "Height (m)")
linearRegression(dfMlj, "Men's Long Jump", "Distance (m)")
linearRegression(dfM100, "Men's 100m Race", "Time (seconds)")
linearRegression(dfM1500, "Men's 1500m Race", "Time (seconds)")
linearRegression(dfW1500, "Women's 1500m Race", "Time (seconds)")
linearRegression(dfOlympic, "Men's Olympic Records", "Time (seconds)")


#%% I don't believe linear regression best fits this data and will explore further
linearRegression(WMaraFast, "Women's Marathon Fastest Times", "Time (seconds)")
linearRegression(MMaraFast, "Men's Marathon Fastest Times", "Time (seconds)")
linearRegression(dfW100, "Women's 100m Race", "Time (seconds)")
linearRegression(MMaraWR, "Men's Marathon WR Times", "Time (seconds)")
linearRegression(WMaraWR, "Women's Marathon WR Times", "Time (seconds)")


#%% Plots
Plot(dfWhj, "Women's High Jump", "Height (m)")
Plot(dfWlj, "Women's Long Jump", "Distance (m)")
Plot(dfMhj, "Men's High Jump", "Height (m)")
Plot(dfMlj, "Men's Long Jump", "Distance (m)")
Plot(dfM100, "Men's 100m Race", "Time (seconds)")
Plot(dfW100, "Women's 100m Race", "Time (seconds)")
Plot(dfM1500, "Men's 1500m Race", "Time (seconds)")
Plot(dfW1500, "Women's 1500m Race", "Time (seconds)")
Plot(dfOlympic, "Men's Olympic Records", "Time (seconds)")
Plot(MMaraWR, "Men's Marathon WR Times", "Time (seconds)")
Plot(WMaraWR, "Women's Marathon WR Times", "Time (seconds)")
Plot(WMaraFast, "Women's Marathon Fastest Times", "Time (seconds)")
Plot(MMaraFast, "Men's Marathon Fastest Times", "Time (seconds)")


#%%Comparison functions
CompareMenandWomenAllEventsPlot()
ComparePlot(dfM100, dfW100, "Men's and Women's 100m Race Progression over the Years", 'percentage')
ComparePlot(dfM1500, dfW1500, "Men's and Women's 1500m Race Progression over the Years", 'percentage')
ComparePlot(dfMhj, dfWhj, "Men's and Women's High Jump Progression over the Years", 'percentage')
ComparePlot(dfMlj, dfWlj, "Men's and Women's Long Jump Progression over the Years", 'percentage')
ComparePlot(MMaraWR, WMaraWR, "Men's and Women's Marathon Record Progression over the Years", 'percentage')


#%% Exploratory analysis
def polynomialRegression(df, eventName, valueColumn):
    if df.equals(WMaraWR):  # This data point occurs 40 years before the next one, so it ruins the regression
        df.drop(index=0, inplace=True)
    x = df['Year'].values.reshape(-1, 1)
    y = df[valueColumn].values.reshape(-1, 1)

    poly1 = PolynomialFeatures(degree=1, include_bias=False)  # Linear model
    poly2 = PolynomialFeatures(degree=2, include_bias=False)  # Quadratic model
    poly5 = PolynomialFeatures(degree=5, include_bias=False)  # Quintic

    # Design Matrices without 1s columns
    poly1_ind_var = poly1.fit_transform(x.reshape(-1, 1))
    poly2_ind_var = poly2.fit_transform(x.reshape(-1, 1))
    poly5_ind_var = poly5.fit_transform(x.reshape(-1, 1))

    # Includes Beta_0
    poly_reg1 = LinearRegression(fit_intercept=True)
    poly_reg2 = LinearRegression(fit_intercept=True)
    poly_reg5 = LinearRegression(fit_intercept=True)

    # Fitting regression models
    poly_reg1.fit(poly1_ind_var, y)
    poly_reg2.fit(poly2_ind_var, y)
    poly_reg5.fit(poly5_ind_var, y)

    # Predicted values for each regression model
    y_predicted_1 = poly_reg1.predict(poly1_ind_var)
    y_predicted_2 = poly_reg2.predict(poly2_ind_var)
    y_predicted_5 = poly_reg5.predict(poly5_ind_var)

    # Calculating mean squared errors and r2 values before plot is displayed
    linear_mse = sklearn.metrics.mean_squared_error(y, y_predicted_1)
    quadratic_mse = sklearn.metrics.mean_squared_error(y, y_predicted_2)
    quintic_mse = sklearn.metrics.mean_squared_error(y, y_predicted_5)
    
    # Seperation in terminal
    print("")
    print(eventName)
    print("")
    # Prints the best fitting MSE
    if linear_mse < quadratic_mse and linear_mse < quintic_mse:
        print(f"The polynomial with the lowest MSE of {linear_mse} was the linear model")
    elif quadratic_mse < linear_mse and quadratic_mse < quintic_mse:
        print(f"The polynomial with the lowest MSE of {quadratic_mse} was the quadratic model")        
    elif quintic_mse < quadratic_mse and quintic_mse < linear_mse:
        print(f"The polynomial with the lowest MSE of {quintic_mse} was the quintic model")
    else:
        print(f"Multiple MSE Values are equal, the linear MSE = {linear_mse}")
        print(f"the quadratic MSE = {quadratic_mse},")
        print(f"the quintic MSE = {quintic_mse}")
        
    linear_r2 = sklearn.metrics.r2_score(y, y_predicted_1)
    quadratic_r2 = sklearn.metrics.r2_score(y, y_predicted_2)
    quintic_r2 = sklearn.metrics.r2_score(y, y_predicted_5)
    LinearDiff = 1 - abs(linear_r2)
    QuadraticDiff = 1 - abs(quadratic_r2)
    QuinticDiff = 1 - abs(quintic_r2)
    
    #prints the best fitting polynomial and its r2 value
    if LinearDiff < QuadraticDiff and LinearDiff < QuinticDiff:
        print(f"The polynomial that had an R squared value closest to 1 was the Linear Model, with R squared value {linear_r2}")
    elif QuadraticDiff < LinearDiff and QuadraticDiff < QuinticDiff:
        print(f"The polynomial that had an R squared value closest to 1 was the Quadratic Model, with R squared value {quadratic_r2}")
    elif QuinticDiff < LinearDiff and QuinticDiff < QuadraticDiff:
        print(f"The polynomial that had an R squared value closest to 1 was the Quintic Model, with R squared value {quintic_r2}")
    else:
        print(f"Multiple R Squared Values are equal, the linear R^2 = {linear_r2}")
        print(f"the Quadratic R^2 = {quadratic_r2}")
        print(f"the Quintic R^2 = {quintic_r2}")

    # Plots the regression models
    plt.scatter(x, y, label='DataPoints')
    plt.title(eventName)
    plt.plot(x, y_predicted_1,'Green', label = "Linear Polynomial Fit")
    plt.plot(x, y_predicted_2,'Purple', label = "Quadratic Polynomial Fit")
    plt.plot(x, y_predicted_5,'Orange', label = "Quintic Polynomial Fit")
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.legend()
    plt.show()

# Plots for Polynomial Regressions
polynomialRegression(WMaraFast, "Women's Marathon Fastest Times", "Time (seconds)")
polynomialRegression(MMaraFast, "Men's Marathon Fastest Times", "Time (seconds)")
polynomialRegression(dfW100, "Women's 100m Race", "Time (seconds)")
polynomialRegression(MMaraWR, "Men's Marathon WR Times", "Time (seconds)")
polynomialRegression(WMaraWR, "Women's Marathon WR Times", "Time (seconds)")

# Analysing the Progression of BMI, weight and height trends in the Olympic
dfOlympicE = MMaraWR = pd.read_excel(pathAthletics, sheet_name="men's 100m", header=1)

columnsToDropOlympicE = ['winner', 'Unnamed: 6', 'Athlete', 'Time (seconds)', 'Date']
ColumnsDropper(dfOlympicE, columnsToDropOlympicE)  # Drops columns array
dfOlympicE.rename(inplace=True, columns={'year': 'Year',
                                         'height (metres)': 'Height (m)',
                                         'weight (kg)': 'Weight (kg)',
                                         'BMI (Body Mass Index)': 'BMI',
                                         'time': 'Time (seconds)'})

polynomialRegression(dfOlympicE, "BMI Progression Throughout the Years for the Men's Olympics", 'BMI')
polynomialRegression(dfOlympicE, "Weight Progression Throughout the Years for the Men's Olympics", "Weight (kg)")
polynomialRegression(dfOlympicE, "Height Progression Throughout the Years for the Men's Olympics", "Height (m)")

# Comparing Times against BMI
def xAgainstYPlot(df, xAxis, yAxis):
    xAxisData = df[xAxis]
    yAxisData = df[yAxis]
    plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
    plt.scatter(xAxisData, yAxisData, label='DataPoints')
    #plt.plot(xAxisData, yAxisData, marker='o', linestyle='-')  # You can customize marker and linestyle
    plt.title(f'{xAxis} against {yAxis} Plot')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.grid(True)  # Add gridlines if desired
    plt.legend()
    plt.show()

xAgainstYPlot(dfOlympicE, 'BMI', 'Time (seconds)')

avgRateOfImprovement(dfWhj, "Women's High Jump", 'Height (m)', True)
avgRateOfImprovement(dfMhj, "Men's High Jump", 'Height (m)', True)
avgRateOfImprovement(dfWlj, "Women's Long Jump", 'Distance (m)', True)
avgRateOfImprovement(dfMlj, "Men's Long Jump", 'Distance (m)', True)
avgRateOfImprovement(dfOlympic, "Men's Olympic 100m Records", 'Time (seconds)', False)
avgRateOfImprovement(dfM100, "Men's 100m", 'Time (seconds)', False)
avgRateOfImprovement(dfW100, "Women's 100m", "Time (seconds)", False)
avgRateOfImprovement(dfM1500, "Men's 1500m", "Time (seconds)", False)
avgRateOfImprovement(MMaraWR, "Men's Marathon World Record Times", "Time (seconds)", False)
avgRateOfImprovement(WMaraWR, "women's Marathon World Record Times", "Time (seconds)", False)
avgRateOfImprovement(MMaraFast, "Men's Fastest Marathon Times", "Time (seconds)", False)
avgRateOfImprovement(WMaraFast, "Women's Fastest Marathon Times", "Time (seconds)", False)
LongestTimeBetweenRecord(dfWhj, "Women's High Jump", "Height (m)", True)