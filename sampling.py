from collections import defaultdict
from math import floor
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from custom_types import RaspberryPi, Attack
from tabulate import tabulate

# TODO:
#   once data collection is completed:
#   merge all the data into one master csv with additional columns for device and attack
#   and refactor this accordingly
#   also remove deprecated stuff
#   and integrate the second pi4_2gb (easier when data is merged before!)
#   check whether further processing valuable


class_map_binary: Dict[Attack, int] = defaultdict(lambda: 1, {
    Attack.NORMAL: 0,
    Attack.NORMAL_V2: 0
})

class_map_multi: Dict[Attack, int] = defaultdict(lambda: 0, {
    Attack.DELAY: 1,
    Attack.DISORDER: 2,
    Attack.FREEZE: 3,
    Attack.HOP: 4,
    Attack.MIMIC: 5,
    Attack.NOISE: 6,
    Attack.REPEAT: 7,
    Attack.SPOOF: 8
})

# TODO: make pi4-2gb-1 and pi4-2gb-2 and add file paths (black and white covers!)
data_file_paths: Dict[RaspberryPi, Dict[Attack, str]] = {
    RaspberryPi.PI3_2GB: {
        Attack.NORMAL: "data/ras-3-2gb/samples_normal_2021-06-18-15-59_50s",
        Attack.NORMAL_V2: "data/ras-3-2gb/samples_normal_v2_2021-06-23-16-54_50s",
        Attack.DELAY: "data/ras-3-2gb/samples_delay_2021-07-01-08-30_50s",
        Attack.DISORDER: "data/ras-3-2gb/samples_disorder_2021-06-30-23-54_50s",
        Attack.FREEZE: "data/ras-3-2gb/samples_freeze_2021-07-01-14-11_50s",
        Attack.HOP: "data/ras-3-2gb/samples_hop_2021-06-29-23-23_50s",
        Attack.MIMIC: "data/ras-3-2gb/samples_mimic_2021-06-30-10-33_50s",
        Attack.NOISE: "data/ras-3-2gb/samples_noise_2021-06-30-19-44_50s",
        Attack.REPEAT: "data/ras-3-2gb/samples_repeat_2021-07-01-20-00_50s",
        Attack.SPOOF: "data/ras-3-2gb/samples_spoof_2021-06-30-14-49_50s"
    },
    RaspberryPi.PI4_2GB: {
        Attack.NORMAL: "data/ras-4-black/samples_normal_2021-07-11-22-19_50s",
        Attack.NORMAL_V2: "data/ras-4-black/samples_normal_v2_2021-07-17-15-38_50s",
        Attack.DELAY: "data/ras-4-black/samples_delay_2021-06-30-14-03_50s",
        Attack.DISORDER: "data/ras-4-black/samples_disorder_2021-06-30-09-44_50s",
        Attack.FREEZE: "data/ras-4-black/samples_freeze_2021-06-29-22-50_50s",
        Attack.HOP: "data/ras-4-black/samples_hop_2021-06-30-18-24_50s",
        Attack.MIMIC: "data/ras-4-black/samples_mimic_2021-06-29-18-35_50s",
        Attack.NOISE: "data/ras-4-black/samples_noise_2021-06-29-14-20_50s",
        Attack.REPEAT: "data/ras-4-black/samples_repeat_2021-06-28-23-52_50s",
        Attack.SPOOF: "data/ras-4-black/samples_spoof_2021-06-28-19-34_50s",
    },
    RaspberryPi.PI4_4GB: {
        Attack.NORMAL: "data/ras-4-4gb/samples_normal_2021-07-09-09-56_50s",
        Attack.NORMAL_V2: "data/ras-4-4gb/samples_normal_v2_2021-07-13-10-43_50s",
        Attack.DELAY: "data/ras-4-4gb/samples_delay_2021-07-01-08-36_50s",
        Attack.DISORDER: "data/ras-4-4gb/samples_disorder_2021-06-30-23-57_50s",
        Attack.FREEZE: "data/ras-4-4gb/samples_freeze_2021-07-01-14-13_50s",
        Attack.HOP: "data/ras-4-4gb/samples_hop_2021-06-29-23-25_50s",
        Attack.MIMIC: "data/ras-4-4gb/samples_mimic_2021-06-30-10-00_50s",
        Attack.NOISE: "data/ras-4-4gb/samples_noise_2021-06-30-19-48_50s",
        Attack.REPEAT: "data/ras-4-4gb/samples_repeat_2021-07-01-20-06_50s",
        Attack.SPOOF: "data/ras-4-4gb/samples_spoof_2021-06-30-14-54_50s"
    },
}


class DataSampler:

    @staticmethod
    def __pick_from_all_data(all_data: pd.DataFrame, device: RaspberryPi, attacks: Dict[Attack, int],
                             label_dict: Dict[Attack, int], pick_ratios: Dict[str, float]) -> \
            Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        data_x, data_y = None, None
        for attack in attacks:
            df = all_data.loc[(all_data['attack'] == attack.value) & (all_data['device'] == device.value)]
            key = device.value + "-" + attack.value
            if pick_ratios[key] < 1.:
                sampled = df.sample(n=floor(pick_ratios[key] * attacks[attack]))
                all_data = pd.concat([sampled, all_data]).drop_duplicates(keep=False)
                n_to_pick = floor(1. / pick_ratios[key] * attacks[attack])
                sampled = sampled.sample(n=n_to_pick, replace=True)
            else:
                sampled = df.sample(n=attacks[attack])
                all_data = pd.concat([sampled, all_data]).drop_duplicates(keep=False)

            if data_x is None:
                data_x = sampled.drop(['attack', 'device'], axis=1).to_numpy()
            else:
                data_x = np.concatenate((data_x, sampled.drop(['attack', 'device'], axis=1).to_numpy()))

            sampled_y = np.array([label_dict[attack]] * (
                floor(1. / pick_ratios[key] * attacks[attack]) if pick_ratios[key] < 1. else attacks[attack]))
            if data_y is None:
                data_y = sampled_y
            else:
                data_y = np.concatenate((data_y, sampled_y))
        data_y = data_y.reshape((len(data_y), 1))
        return all_data, data_x, data_y

    # TODO: once merged: check if file exists, if yes: pd.read_csv, else this style
    @staticmethod
    def __parse_all_files_to_df() -> pd.DataFrame:
        full_df = pd.DataFrame()
        for device in data_file_paths:
            for attack in data_file_paths[device]:
                df = pd.read_csv(data_file_paths[device][attack])
                # filter for measurements where the device was connected
                df = df[df['connectivity'] == 1]
                # remove model-irrelevant columns
                df = df.drop(["time", "timestamp", "seconds", "connectivity"], axis=1)
                df['device'] = device.value
                df['attack'] = attack.value
                full_df = pd.concat([full_df, df])
        #full_df.to_csv('./data/all.csv', index_label=False)
        return full_df

    @staticmethod
    def show_data_availability():
        all_data = DataSampler.__parse_all_files_to_df()
        drop_cols = [col for col in list(all_data) if col not in ['device', 'attack', 'alarmtimer:alarmtimer_fired']]
        print(tabulate(
            all_data.drop(drop_cols, axis=1).rename(columns={'alarmtimer:alarmtimer_fired': 'count'}).groupby(
                ['device', 'attack'], as_index=False).count(), tablefmt="pretty"))

    @staticmethod
    def get_all_clients_data(
            train_devices: List[Tuple[RaspberryPi, Dict[Attack, int], Dict[Attack, int]]],
            test_devices: List[Tuple[RaspberryPi, Dict[Attack, int]]],
            multi_class=False) -> \
            Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:

        assert len(train_devices) > 0 and len(
            test_devices) > 0, "Need to provide at least one train and one test device!"

        all_data = DataSampler.__parse_all_files_to_df()

        # Dictionaries that hold total request: e. g. we want 500 train data for a pi3 and delay
        # but may only have 100 -> oversample and prevent overlaps
        total_data_request_count_train: Dict[str, int] = defaultdict(lambda: 0)
        remaining_data_available_count: Dict[str, int] = defaultdict(lambda: 0)

        # determine how to label data
        label_dict = class_map_multi if multi_class else class_map_binary

        for device, attacks, validation_attacks in train_devices:
            for attack in attacks:
                total_data_request_count_train[device.value + "-" + attack.value] += attacks[attack]

        train_sets = []
        validation_sets = []
        test_sets = []

        # pick test sets
        for device, test_attacks in test_devices:
            all_data, test_x, test_y = DataSampler.__pick_from_all_data(all_data, device, test_attacks, label_dict,
                                                                        defaultdict(lambda: 1))
            test_sets.append((test_x, test_y))

        # pick validation sets: same as test sets -> in refactoring can be merged
        for device, _, val_attacks in train_devices:
            all_data, val_x, val_y = DataSampler.__pick_from_all_data(all_data, device, val_attacks, label_dict,
                                                                      defaultdict(lambda: 1))
            validation_sets.append((val_x, val_y))

        for __i, row in all_data.groupby(['device', 'attack']).count().iterrows():
            remaining_data_available_count[row.name[0] + "-" + row.name[1]] += row.cs

        train_ratio_dict = {}
        for key in total_data_request_count_train:
            train_ratio_dict[key] = float(remaining_data_available_count[key]) / total_data_request_count_train[key]

        # pick and sample train sets
        for device, attacks, _ in train_devices:
            all_data, train_x, train_y = DataSampler.__pick_from_all_data(all_data, device, attacks, label_dict,
                                                                          train_ratio_dict)
            train_sets.append((train_x, train_y))

        return [(x, y, validation_sets[idx][0], validation_sets[idx][1]) for
                idx, (x, y) in enumerate(train_sets)], [(x, y) for x, y in test_sets]

    @staticmethod
    def scale(train_devices: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
              test_devices: List[Tuple[np.ndarray, np.ndarray]], central=False) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
        train_scaled = []
        test_scaled = []
        if central:
            assert len(train_devices) == 1, "Only single training device allowed in central mode!"
            scaler: StandardScaler = StandardScaler()
            scaler.fit(train_devices[0][0])
            for x_train, y_train, x_val, y_val in train_devices:
                train_scaled.append((scaler.transform(x_train), y_train, scaler.transform(x_val), y_val))
            for x_test, y_test in test_devices:
                test_scaled.append((scaler.transform(x_test), y_test))
        else:
            scalers: List[MinMaxScaler] = []
            for x_train, y_train, x_val, y_val in train_devices:
                scaler: MinMaxScaler = MinMaxScaler(clip=True)
                scaler.fit(x_train)
                scalers.append(scaler)
            final_scaler = MinMaxScaler(clip=True)
            final_scaler.min_ = np.stack([s.min_ for s in scalers], axis=1).mean(axis=1)
            final_scaler.scale_ = np.stack([s.scale_ for s in scalers], axis=1).mean(axis=1)
            for x_train, y_train, x_val, y_val in train_devices:
                train_scaled.append((final_scaler.transform(x_train), y_train, final_scaler.transform(x_val), y_val))
            for x_test, y_test in test_devices:
                test_scaled.append((final_scaler.transform(x_test), y_test))

        return train_scaled, test_scaled
