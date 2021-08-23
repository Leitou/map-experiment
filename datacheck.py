import os
import pandas as pd
from sampling import data_file_paths
from custom_types import Attack, RaspberryPi
import matplotlib.pyplot as plt


def parse_all_files_to_df_raw() -> pd.DataFrame:
    if os.path.isfile("./data/all_raw.csv"):
        return pd.read_csv("./data/all_raw.csv")
    full_df = pd.DataFrame()
    for device in data_file_paths:
        for attack in data_file_paths[device]:
            df = pd.read_csv(data_file_paths[device][attack])
            # filter for measurements where the device was connected
            df = df[df['connectivity'] == 1]
            #remove model-irrelevant columns
            #df = df.drop(["time", "timestamp", "seconds", "connectivity"], axis=1)
            df['device'] = device.value
            df['attack'] = attack.value
            full_df = pd.concat([full_df, df])
    full_df.to_csv('./data/raw_all.csv', index_label=False)
    return full_df

# for each of the devices plot their data for all the different options
# 4 devices, for each 10 monitoring programs -> total 40

# TODO: first just plot a random feature at some index
all_data = parse_all_files_to_df_raw()

d = {}
for name, group in all_data.groupby(['device', 'attack']):
    ft = group.columns[17]
    print("device: ", name[0], "- feature: ", ft)

    if name[1] == "normal":
        group.plot(x="time", y=f'{ft}', kind='scatter', color="b")
        plt.title(f"{name[0] + '-' + name[1]}")
        plt.show()
    elif name[1] == "normal_v2":
        group.plot(x="time", y=f'{ft}', kind='scatter', color="r")
        plt.title(f"{name[0] + '-' + name[1]}")
        plt.show()









#
#     d['group_' + str(name)] = group
# for __i, row in all_data.groupby(['device', 'attack']).count().iterrows():