def fill_bin_num(dataframe, feature, bin_feature, bin_size, stat_measure, min_bin=None, max_bin=None, default_val='No'):
    if min_bin is None:
        min_bin = dataframe[bin_feature].min()
    if max_bin is None:
        max_bin = dataframe[bin_feature].max()
    new_dataframe = dataframe.copy()
    df_meancat = pd.DataFrame(columns=['interval', 'stat_measure'])
    for num_bin, subset in dataframe.groupby(pd.cut(dataframe[bin_feature], np.arange(min_bin, max_bin+bin_size, bin_size), include_lowest=True)):
        if stat_measure is 'mean':
            row = [num_bin, subset[feature].mean()]
        elif stat_measure is 'mode':
            mode_ar = subset[feature].mode().values
            if len(mode_ar) > 0:
                row = [num_bin, mode_ar[0]]
            else:
                row = [num_bin, default_val]
        else:
            raise Exception('Unknown statistical measure: ' + stat_measure)
        df_meancat.loc[len(df_meancat)] = row
    for index, row_df in dataframe[dataframe[feature].isna()].iterrows():
        for _, row_meancat in df_meancat.iterrows():
            if row_df[bin_feature] in row_meancat['interval']:
                new_dataframe.at[index, feature] = row_meancat['stat_measure']
    return new_dataframe


def make_dummy_cols(dataframe, column, prefix, drop_dummy):
    dummy = pd.get_dummies(dataframe[column], prefix=prefix)
    dummy = dummy.drop(columns=prefix+'_'+drop_dummy)
    dataframe = pd.concat([dataframe, dummy], axis=1)
    dataframe = dataframe.drop(columns=column)
    return dataframe


def cleaning(dataframe_raw):
    dataframe = dataframe_raw.copy()

    dataframe = dataframe.set_index('ID')

    dataframe.loc[(dataframe['Age']<=13) & (dataframe['Education'].isna()), 'Education'] = 'Lower School/Kindergarten'
    dataframe.loc[(dataframe['Age']==14) & (dataframe['Education'].isna()), 'Education'] = '8th Grade'
    dataframe.loc[(dataframe['Age']<=17) & (dataframe['Education'].isna()), 'Education'] = '9 - 11th Grade'
    dataframe.loc[(dataframe['Age']<=21) & (dataframe['Education'].isna()), 'Education'] = 'High School'
    dataframe['Education'] = dataframe['Education'].fillna('Some College')

    dataframe.loc[(dataframe['Age']<=20) & (dataframe['MaritalStatus'].isna()), 'MaritalStatus'] = 'NeverMarried'
    dataframe.at[dataframe['MaritalStatus'].isna(), 'MaritalStatus'] = 'Married'  # For now this is hardcoded

    dataframe = dataframe.drop(columns=['HHIncome'])

    dataframe.loc[dataframe['HHIncomeMid'].isna(), 'HHIncomeMid'] = dataframe['HHIncomeMid'].mean()

    dataframe.loc[dataframe['Poverty'].isna(), 'Poverty'] = dataframe['Poverty'].mean()

    dataframe.loc[dataframe['HomeRooms'].isna(), 'HomeRooms'] = dataframe['HomeRooms'].mean()

    dataframe.loc[dataframe['HomeOwn'].isna(), 'HomeOwn'] = dataframe['HomeOwn'].mode().values[0]

    dataframe.loc[(dataframe['Work'].isna()) & (dataframe['Education'].isna()) & (dataframe['Age']<=20), 'Work'] = 'NotWorking'

    dataframe.loc[dataframe['Work'].isna(), 'Work'] = dataframe['Work'].mode().values[0]

    dataframe = fill_bin_num(dataframe, 'Weight', 'Age', 2, 'mean')

    dataframe = dataframe.drop(columns=['HeadCirc'])

    for index, row in dataframe.iterrows():
        if np.isnan(row['Height']) and not np.isnan(row['Length']):
            dataframe.at[index, 'Height'] = row['Length']
    dataframe = fill_bin_num(dataframe, 'Height', 'Age', 2, 'mean')

    dataframe = dataframe.drop(columns=['Length'])

    for index, row in dataframe[dataframe['BMI'].isna()].iterrows():
        dataframe.at[index, 'BMI'] = row['Weight'] / ((row['Height']/100)**2)

    dataframe = dataframe.drop(columns='BMICatUnder20yrs')

    dataframe = dataframe.drop(columns='BMI_WHO')

    dataframe = fill_bin_num(dataframe, 'Pulse', 'Age', 10, 'mean')

    dataframe.loc[(dataframe['Age']<10) & (dataframe['BPSysAve'].isna()), 'BPSysAve'] = 105
    dataframe = fill_bin_num(dataframe, 'BPSysAve', 'Age', 5, 'mean', 10)

    dataframe.loc[(dataframe['Age']<10) & (dataframe['BPDiaAve'].isna()), 'BPDiaAve'] = 60
    dataframe = fill_bin_num(dataframe, 'BPDiaAve', 'Age', 5, 'mean', 10)

    dataframe = dataframe.drop(columns='BPSys1')

    dataframe = dataframe.drop(columns='BPDia1')

    dataframe = dataframe.drop(columns='BPSys2')

    dataframe = dataframe.drop(columns='BPDia2')

    dataframe = dataframe.drop(columns='BPSys3')

    dataframe = dataframe.drop(columns='BPDia3')

    dataframe = dataframe.drop(columns=['Testosterone'])

    # Set to 0 for the time being because I cannot find the values
    dataframe.loc[(dataframe['Age']<10) & (dataframe['DirectChol'].isna()), 'DirectChol'] = 0
    dataframe = fill_bin_num(dataframe, 'DirectChol', 'Age', 5, 'mean', 10)

    dataframe.loc[(dataframe['Age']<10) & (dataframe['TotChol'].isna()), 'TotChol'] = 0
    dataframe = fill_bin_num(dataframe, 'TotChol', 'Age', 5, 'mean', 10)

    dataframe = dataframe.drop(columns=['UrineVol1'])

    dataframe = dataframe.drop(columns=['UrineFlow1'])

    dataframe = dataframe.drop(columns=['UrineVol2'])

    dataframe = dataframe.drop(columns=['UrineFlow2'])

    dataframe['Diabetes'] = dataframe['Diabetes'].fillna('No')

    dataframe['DiabetesAge'] = dataframe['DiabetesAge'].fillna(0)

    dataframe.loc[(dataframe['Age']<=12) & (dataframe['HealthGen'].isna()), 'HealthGen'] = 'Good'
    dataframe = fill_bin_num(dataframe, 'HealthGen', 'Age', 5, 'mode', 10)

    dataframe.loc[(dataframe['Age']<=12) & (dataframe['DaysMentHlthBad'].isna()), 'DaysMentHlthBad'] = 0
    dataframe = fill_bin_num(dataframe, 'DaysMentHlthBad', 'Age', 5, 'mean', 10)

    dataframe.loc[(dataframe['Age']<=15) & (dataframe['LittleInterest'].isna()), 'LittleInterest'] = 'None'
    dataframe = fill_bin_num(dataframe, 'LittleInterest', 'Age', 5, 'mode', 15)

    dataframe.loc[(dataframe['Age']<=12) & (dataframe['DaysMentHlthBad'].isna()), 'DaysMentHlthBad'] = 0
    dataframe = fill_bin_num(dataframe, 'DaysMentHlthBad', 'Age', 5, 'mean', 10)

    dataframe['nPregnancies'] = dataframe['nPregnancies'].fillna(0)

    dataframe['nBabies'] = dataframe['nBabies'].fillna(0)

    dataframe['Age1stBaby'] = dataframe['Age1stBaby'].fillna(0)

    dataframe.loc[(dataframe['Age']==0) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 14
    dataframe.loc[(dataframe['Age']<=2) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 12
    dataframe.loc[(dataframe['Age']<=5) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 10
    dataframe.loc[(dataframe['Age']<=10) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 9
    dataframe.loc[(dataframe['Age']<=15) & (dataframe['SleepHrsNight'].isna()), 'SleepHrsNight'] = 8
    dataframe['SleepHrsNight'] = dataframe['SleepHrsNight'].fillna(dataframe_raw['SleepHrsNight'].mean())

    dataframe['SleepTrouble'] = dataframe['SleepTrouble'].fillna(0)

    dataframe.loc[(dataframe['Age']<=4) & (dataframe['PhysActive'].isna()), 'PhysActive'] = 'No'
    dataframe = fill_bin_num(dataframe, 'PhysActive', 'Age', 2, 'mode', 16)
    dataframe['PhysActive'] = dataframe['PhysActive'].fillna('Yes') # Big assumption here. All kids between 4 and 16 are physically active

    dataframe = dataframe.drop(columns=['PhysActiveDays'])

    dataframe = dataframe.drop(columns=['TVHrsDay'])

    dataframe = dataframe.drop(columns=['TVHrsDayChild'])

    dataframe = dataframe.drop(columns=['CompHrsDay'])

    dataframe = dataframe.drop(columns=['CompHrsDayChild'])

    dataframe.loc[(dataframe['Age']<18) & (dataframe['Alcohol12PlusYr'].isna()), 'Alcohol12PlusYr'] = 'No'
    dataframe = fill_bin_num(dataframe, 'Alcohol12PlusYr', 'Age', 5, 'mode', 18)

    dataframe.loc[(dataframe['Age']<18) & (dataframe['AlcoholDay'].isna()), 'AlcoholDay'] = 0
    dataframe = fill_bin_num(dataframe, 'AlcoholDay', 'Age', 5, 'mode', 18)

    dataframe.loc[(dataframe['Age']<18) & (dataframe['AlcoholYear'].isna()), 'AlcoholYear'] = 0
    dataframe = fill_bin_num(dataframe, 'AlcoholYear', 'Age', 5, 'mode', 18)

    dataframe.loc[(dataframe['Age']<20) & (dataframe['SmokeNow'].isna()), 'SmokeNow'] = 'No'
    dataframe = fill_bin_num(dataframe, 'SmokeNow', 'Age', 5, 'mode', 20)

    dataframe['Smoke100'] = dataframe['Smoke100'].fillna('No')

    dataframe['Smoke100n'] = dataframe['Smoke100'].fillna('No')

    dataframe.loc[(dataframe['SmokeNow']=='No') & (dataframe['SmokeAge'].isna()), 'SmokeAge'] = 0
    dataframe = fill_bin_num(dataframe, 'SmokeAge', 'Age', 5, 'mean', 20)

    dataframe.loc[(dataframe['Age']<18) & (dataframe['Marijuana'].isna()), 'Marijuana'] = 'No'
    dataframe.loc[(dataframe['Marijuana'].isna()) & (dataframe['SmokeNow']=='No'), 'Marijuana'] = 'No'
    dataframe = fill_bin_num(dataframe, 'Marijuana', 'Age', 5, 'mode', 20)

    dataframe.loc[(dataframe['Marijuana']=='No') & (dataframe['AgeFirstMarij'].isna()), 'AgeFirstMarij'] = 0
    dataframe = fill_bin_num(dataframe, 'AgeFirstMarij', 'Age', 5, 'mean', 20)

    dataframe.loc[(dataframe['Marijuana']=='No') & (dataframe['RegularMarij'].isna()), 'RegularMarij'] = 'No'
    dataframe = fill_bin_num(dataframe, 'RegularMarij', 'Age', 5, 'mode', 20)

    dataframe.loc[(dataframe['RegularMarij']=='No') & (dataframe['AgeRegMarij'].isna()), 'AgeRegMarij'] = 0
    dataframe = fill_bin_num(dataframe, 'AgeRegMarij', 'Age', 5, 'mean', 20)

    dataframe.loc[(dataframe['Age']<18) & (dataframe['HardDrugs'].isna()), 'HardDrugs'] = 'No'
    dataframe = fill_bin_num(dataframe, 'HardDrugs', 'Age', 5, 'mode', 18)

    mode_sex_age = dataframe['SexAge'].mode()[0]
    dataframe.loc[(dataframe['Age']<=mode_sex_age) & (dataframe['SexEver'].isna()), 'SexEver'] = 'No'
    dataframe['SexEver'] = dataframe['SexEver'].fillna('Yes')

    dataframe.loc[(dataframe['SexEver']=='No') & (dataframe['SexAge'].isna()), 'SexAge'] = 0
    dataframe.loc[(dataframe['SexAge'].isna() & (dataframe['Age']<mode_sex_age)), 'SexAge'] = dataframe.loc[(dataframe['SexAge'].isna() & (dataframe['Age']<mode_sex_age)), 'Age']
    dataframe['SexAge'] = dataframe['SexAge'].fillna(mode_sex_age)

    dataframe.loc[(dataframe['SexEver']=='No') & (dataframe['SexNumPartnLife'].isna()), 'SexNumPartnLife'] = 0
    dataframe = fill_bin_num(dataframe, 'SexNumPartnLife', 'Age', 5, 'mean')
    dataframe['SexNumPartnLife'] = dataframe_raw.loc[(dataframe_raw['Age'] >= 60) & (dataframe_raw['Age'] <= 70), 'SexNumPartnLife'].mode()[0] # Missing values for the elderly. Assumed that lifetime sex partners do not increase after 60.

    dataframe.loc[(dataframe['SexEver']=='No') & (dataframe['SexNumPartYear'].isna()), 'SexNumPartYear'] = 0
    dataframe = fill_bin_num(dataframe, 'SexNumPartYear', 'Age', 10, 'mean')
    dataframe['SexNumPartYear'] = dataframe['SexNumPartYear'].fillna(0)

    dataframe = dataframe.drop(columns=['SameSex'])

    dataframe = dataframe.drop(columns=['SexOrientation']) # Maybe this should not be dropped

    dataframe['PregnantNow'] = dataframe['PregnantNow'].fillna('No')


    # Making dummy variables
    dataframe['male'] = 1*(dataframe['Gender'] ==  'male')
    dataframe = dataframe.drop(columns=['Gender'])

    dataframe['white'] = np.where(dataframe['Race1'] == 'white',1,0)
    dataframe = dataframe.drop(columns=['Race1'])

    dataframe = make_dummy_cols(dataframe, 'Education', 'education', '8th Grade')

    dataframe = make_dummy_cols(dataframe, 'MaritalStatus', 'maritalstatus', 'Separated')

    dataframe = make_dummy_cols(dataframe, 'HomeOwn', 'homeown', 'Other')

    dataframe = make_dummy_cols(dataframe, 'Work', 'work', 'Looking')

    dataframe['Diabetes'] = np.where(dataframe['Diabetes'] == 'Yes',1,0)

    dataframe = make_dummy_cols(dataframe, 'HealthGen', 'healthgen', 'Poor')

    dataframe = make_dummy_cols(dataframe, 'LittleInterest', 'littleinterest', 'None')

    dataframe = make_dummy_cols(dataframe, 'Depressed', 'depressed', 'None')

    dataframe['SleepTrouble'] = np.where(dataframe['SleepTrouble'] == 'Yes',1,0)

    dataframe['PhysActive'] = np.where(dataframe['PhysActive'] == 'Yes',1,0)


    dataframe['Alcohol12PlusYr'] = np.where(dataframe['Alcohol12PlusYr'] == 'Yes',1,0)

    dataframe['SmokeNow'] = np.where(dataframe['SmokeNow'] == 'Yes',1,0)

    dataframe['Smoke100'] = np.where(dataframe['Smoke100'] == 'Yes',1,0)

    dataframe['Smoke100n'] = np.where(dataframe['Smoke100n'] == 'Yes',1,0)

    dataframe['Marijuana'] = np.where(dataframe['Marijuana'] == 'Yes',1,0)

    dataframe['RegularMarij'] = np.where(dataframe['RegularMarij'] == 'Yes',1,0)

    dataframe['HardDrugs'] = np.where(dataframe['HardDrugs'] == 'Yes',1,0)

    dataframe['SexEver'] = np.where(dataframe['SexEver'] == 'Yes',1,0)

    # dataframe['Heterosexual'] = np.where(dataframe['SexOrientation'] == 'Heterosexual',1,0)
    # dataframe = dataframe.drop(columns=['SexOrientation'])

    dataframe['PregnantNow'] = np.where(dataframe['PregnantNow'] == 'Yes',1,0)

    return dataframe
