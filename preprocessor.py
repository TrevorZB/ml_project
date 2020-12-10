#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# atlantic ocean hurricanes
atlantic = pd.read_csv('atlantic.csv')
hurricanes = atlantic[['ID', 'Maximum Wind']]

# hurricanes occur on multiple rows, group by hurricane ID, find max wind of each hurricane
hurricanes = hurricanes.groupby(hurricanes['ID'])['Maximum Wind'].max().reset_index(name='Max_Wind')

# dates with ID's of hurricanes in the hurricane dataframe
final_data = atlantic[['Date', 'ID']]

# only use data from 1852 and later
final_data = final_data[final_data['Date'].astype(str).str[0:4].astype(int) > 1851]

# group by month, turn hurricanes ids into a list for each month
final_data = final_data.groupby(final_data.Date.astype(str).str[0:6])['ID'].apply(list).reset_index(name='Hurricane_IDs')

# removes duplicate hurricanes
final_data['Hurricane_IDs'] = final_data['Hurricane_IDs'].map(np.unique)

final_data.columns = ['dt', 'Hurricane_IDs']

# global surface temperatures
df = pd.read_csv('global_temps.csv')

# grab final_data and temperatures
df = df[['dt', 'LandAverageTemperature', 'LandMinTemperature', 'LandMaxTemperature', 'LandAndOceanAverageTemperature']]

# only get rows that are 1852 and later
df = df[df['dt'].astype(str).str[0:4].astype(int) > 1851]

# modify the dt column to join with the hurricane table
df['dt'] = df['dt'].astype(str).str[0:4] + df['dt'].astype(str).str[5:7]

# merge the hurricane and global temperature data frames
final_data = pd.merge(final_data, df, on='dt')

# average temperatures by country
usa_temp = pd.read_csv('GlobalLandTemperaturesByCountry.csv')

# grab dates, tempeartures, and country
usa_temp = usa_temp[['dt', 'AverageTemperature', 'Country']]

# grab united states rows
usa_temp = usa_temp[usa_temp['Country'] == 'United States']

# rename the temp column
usa_temp.rename(columns={'AverageTemperature': 'AvgTempUSA'}, inplace=True)

# 1852 or later
usa_temp = usa_temp[usa_temp['dt'].astype(str).str[0:4].astype(int) > 1851]

# get rid of country column
usa_temp = usa_temp[['dt', 'AvgTempUSA']]

# modify the dt column to join with the aggregated table
usa_temp['dt'] = usa_temp['dt'].astype(str).str[0:4] + usa_temp['dt'].astype(str).str[5:7]

# merge usa temp into the overall table
final_data = pd.merge(final_data, usa_temp, on='dt')

# average temperatures by state
state_temps = pd.read_csv('GlobalLandTemperaturesByState.csv')

# only usa states
state_temps = state_temps[state_temps['Country'] == 'United States']

# 1852 or later
state_temps = state_temps[state_temps['dt'].astype(str).str[0:4].astype(int) > 1851]

# Top 5 american states affected by hurricanes
states = ['Florida', 'Texas', 'North Carolina', 'Louisiana', 'South Carolina']
state_temps = state_temps[state_temps['State'].isin(states)]

# grab relevant columns
state_temps = state_temps[['dt', 'AverageTemperature', 'State']]

# rename columns
state_temps.rename(columns={'AverageTemperature': 'AvgTempState'}, inplace=True)

# modify the dt column to join with the aggregated table
state_temps['dt'] = state_temps['dt'].astype(str).str[0:4] + state_temps['dt'].astype(str).str[5:7]

# merge for temp table
temp = pd.merge(final_data, state_temps, on='dt')

# group state and temp together
d = temp.groupby('dt')[['State', 'AvgTempState']].apply(lambda g: g.values.tolist()).reset_index(name='StateTemps')

# format state columns correctly
d['Fl'] = d['StateTemps'].str[0]
d['AvgTempFlorida'] = d['Fl'].str[1]

d['Lo'] = d['StateTemps'].str[1]
d['AvgTempLouisiana'] = d['Lo'].str[1]

d['NC'] = d['StateTemps'].str[2]
d['AvgTempNorthCarolina'] = d['NC'].str[1]

d['SC'] = d['StateTemps'].str[3]
d['AvgTempSouthCarolina'] = d['SC'].str[1]

d['Tex'] = d['StateTemps'].str[4]
d['AvgTempTexas'] = d['Tex'].str[1]

# grab only the state columns
d = d[['dt', 'AvgTempFlorida', 'AvgTempLouisiana', 'AvgTempNorthCarolina', 'AvgTempSouthCarolina', 'AvgTempTexas']]

# merge states into overall table
final_data = pd.merge(final_data, d, on='dt')

# calculate ACE value for each entry in the table
sums = []
for entry in final_data['Hurricane_IDs']:
    max_winds = []
    for id in entry:
        max_winds.append(hurricanes.loc[hurricanes['ID'] == id]['Max_Wind'].values[0] ** 2)
    sums.append(sum(max_winds) / 1000)
final_data['ACE'] = sums

# save as a new csv
from os import path, remove
if path.isfile('final_data.csv'):
    remove('final_data.csv')
final_data.to_csv('final_data.csv', index=False)

print('final_data.csv saved')
print('example rows from final_data.csv:')
print(final_data)

