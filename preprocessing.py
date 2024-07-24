import io, os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


# To print the columns of the dataframe without wrapping
pd.set_option('display.expand_frame_repr', False)

# Reads the txt and converts it to csv transforming the dates into timestamps
def txt_to_csv(path):
    df = pd.read_csv(path, sep='\t+', engine='python')
    df.drop(0, inplace=True)
    df.columns = [x.strip().lower().replace(' ', '_') for x in df.columns]
    df['start_time'] = df['start_time'].apply(date_to_timestamp)
    df['end_time'] = df['end_time'].apply(date_to_timestamp)

    return df


# TODO: Take the one that has most
# It discretizes time and merges the two datasets of activities and observations
def merge_dataset_slice(adl, obs, start_date, end_date, length=60):
    first_minute = date_to_timestamp(start_date)
    last_minute = date_to_timestamp(end_date)
    n_sens = max(obs['location']) + 1

    timestamps = []; activities = []; sensors = []; periods = []
    for s in tqdm(range(first_minute, last_minute + 1, length)):
        # Find the activity at time i
        e = s + length - 1
        q = adl.query('@e >= start_time and end_time >= @s')

        # If there is no activity for the current minute, continue
        if q.shape[0] == 0: continue
        activity = q.iloc[0]['activity']

        # Find active sensors at time i
        q = obs.query('@e >= start_time and end_time >= @s')
        sl = q['location'].tolist()
        active_sensors = "".join('1' if x in sl else '0' for x in range(n_sens))

        # Calculate the time of day
        period = day_period(s)

        timestamps.append(s)
        activities.append(activity)
        sensors.append(active_sensors)
        periods.append(period)

    result = pd.DataFrame(
        columns=['timestamp', 'activity', 'sensors', 'period'],
        data = {
            'timestamp': timestamps,
            'activity': activities,
            'sensors': sensors,
            'period': periods,
        }
    )

    return result


# The date appeared
def date_to_timestamp(m):
    return int(datetime.strptime(m.strip(), "%Y-%m-%d %H:%M:%S").timestamp())


# Returns the minute of the day
def day_minute(timestamp):
    return ((timestamp // 60) % (24*60))


# Divides the day into 4 slices
# Returns the fraction of the day to which the timestamp belongs
def day_period(timestamp):
    h = ((timestamp // (60*60)) % 24)
    if h < 6: return 0
    elif h < 12: return 1
    elif h < 18: return 2
    else: return 3


def generate_dataset():
    if not os.path.exists('dataset_csv'): os.makedirs('dataset_csv')
    files = [
        'OrdonezA_ADLs',
        'OrdonezA_Sensors',
        'OrdonezB_ADLs',
        'OrdonezB_Sensors',
    ]

    dfs = {}
    for f in files:
        df = txt_to_csv(f'dataset_costa/{f}.txt')
        df.sort_values(by=['end_time'], inplace=True)

        # Delete inconsistent lines (they end before they start)
        df.drop(df[df['start_time'] > df['end_time']].index, inplace=True)

        # Convert categorial values to integers
        if f.find('ADL') > 0: cols = ['activity']
        else: cols = ['location', 'type', 'place']
        df[cols] = df[cols].apply(lambda x: x.astype('category'))
        if f.find('ADL') > 0:
            activitiesasd = dict(enumerate(df['activity'].cat.categories))
        df[cols] = df[cols].apply(lambda x: x.cat.codes)


        # Save the csv. Just in case
        df.to_csv(f'dataset_csv/{f}.csv', index=False)
        df.reset_index(inplace=True)
        dfs[f] = df


        dataset = []    # list of (merged) datasets to return
        dt = 0          # index of the list

    for f in range(2):
        adl = dfs[files[2 * f]]
        obs = dfs[files[2 * f + 1]]

        # Associate the activity of each sensor with each event that occurred
        # during sensor activity.pa
        start_date = "2011-11-28 00:00:00" if f == 0 else "2012-11-11 00:00:00"
        end_date = "2011-12-11 23:59:59" if f == 0 else "2012-12-02 23:59:59"
        merged = merge_dataset_slice(adl, obs, start_date, end_date)

        # saved the merged datasets in a list which I then return
        dataset[dt] = merged
        dt = dt + 1

        merged.to_csv(f'dataset_csv/Ordonez{"A" if f == 0 else "B"}.csv',
            sep=',', index=False)

    return dataset


# if __name__ == '__main__':
#     generate_dataset()
