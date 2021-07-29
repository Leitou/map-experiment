from collections import defaultdict
from math import floor
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from custom_types import RaspberryPi, Attack

# TODO:
#   once data collection is completed:
#   merge all the data into one master csv with additional columns for device and attack
#   and refactor this accordingly
#   also remove deprecated stuff
#   and integrate the second pi4_2gb (easier when data is merged before!)
#   check whether further processing valuable


# assuming everything not normal is malicious
malicious = defaultdict(lambda: 1)
malicious["normal"] = 0
malicious["normal_v2"] = 0

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
    def get_all_clients_train_data_and_scaler(
            train_devices: List[Tuple[RaspberryPi, Dict[Attack, int], Dict[Attack, int]]],
            test_devices: List[Tuple[RaspberryPi, Dict[Attack, int]]],
            multi_class=False) -> \
            Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:

        assert len(train_devices) > 0 and len(
            test_devices) > 0, "Need to provide at least one train and one test device!"
        # Dictionaries that hold total request: e. g. we want 500 train data for a pi3 and delay
        # but may only have 100 -> oversample and prevent overlaps
        total_data_request_count_train: Dict[str, int] = defaultdict(lambda: 0)
        total_data_request_count_valid: Dict[str, int] = defaultdict(lambda: 0)
        total_data_request_count_test: Dict[str, int] = defaultdict(lambda: 0)
        total_data_available_count: Dict[str, int] = defaultdict(lambda: 0)

        # determine how to label data
        label_dict = class_map_multi if multi_class else class_map_binary

        # pandas data frames for devices and attacks that are requested
        data_frames: Dict[RaspberryPi, Dict[Attack, pd.DataFrame]] = {}

        for device, attacks, validation_attacks in train_devices:
            if device not in data_frames:
                data_frames[device] = {}
            for attack in attacks:
                total_data_request_count_train[device.value + "-" + attack.value] += attacks[attack]
                if attack not in data_frames[device]:
                    data_frames[device][attack] = pd.read_csv(data_file_paths[device][attack])
            for attack in validation_attacks:
                total_data_request_count_valid[device.value + "-" + attack.value] += validation_attacks[attack]
                if attack not in data_frames[device]:
                    data_frames[device][attack] = pd.read_csv(data_file_paths[device][attack])

        for device, attacks in test_devices:
            if device not in data_frames:
                data_frames[device] = {}
            for attack in attacks:
                total_data_request_count_test[device.value + "-" + attack.value] += attacks[attack]
                if attack not in data_frames[device]:
                    data_frames[device][attack] = pd.read_csv(data_file_paths[device][attack])

        for device in data_frames:
            for attack in data_frames[device]:
                df = data_frames[device][attack]
                # filter for connectivity
                df = df[df['connectivity'] == 1]
                # remove model-irrelevant columns
                df = df.drop(["time", "timestamp", "seconds", "connectivity"], axis=1)
                data_frames[device][attack] = df
                total_data_available_count[device.value + "-" + attack.value] = len(df)

        print("Data availability:", dict(total_data_available_count))

        for key in total_data_request_count_test:
            if (total_data_request_count_test[key] + total_data_request_count_valid[key]) > \
                    total_data_available_count[key]:
                raise ValueError(
                    f'Too much data requested for {key}. Please lower sample number! '
                    f'Available: {total_data_available_count[key]}, '
                    f'but requested {total_data_request_count_test[key] + total_data_request_count_valid[key]}')
        for key in total_data_request_count_valid:
            if (total_data_request_count_test[key] + total_data_request_count_valid[key]) > \
                    total_data_available_count[key]:
                raise ValueError(
                    f'Too much data requested for {key}. Please lower sample number! '
                    f'Available: {total_data_available_count[key]}, '
                    f'but requested {total_data_request_count_test[key] + total_data_request_count_valid[key]}')

        train_sets = []
        validation_sets = []
        test_sets = []

        # pick test sets
        for device, attacks in test_devices:
            data_x, data_y = None, None
            for attack in attacks:
                df = data_frames[device][attack]
                sampled = df.sample(n=attacks[attack])
                if data_x is None:
                    data_x = sampled.to_numpy()
                else:
                    data_x = np.concatenate((data_x, sampled.to_numpy()))

                if data_y is None:
                    data_y = np.array([label_dict[attack]] * attacks[attack])
                else:
                    data_y = np.concatenate((data_y, np.array([label_dict[attack]] * attacks[attack])))
                df = pd.concat([df, sampled]).drop_duplicates(keep=False)
                data_frames[device][attack] = df
            data_y = data_y.reshape((len(data_y), 1))
            test_sets.append((data_x, data_y))

        # pick validation sets: same as test sets -> in refactoring can be merged
        for device, _, validation_attacks in train_devices:
            data_valid_x, data_valid_y = None, None
            for attack in validation_attacks:
                df = data_frames[device][attack]
                sampled = df.sample(n=validation_attacks[attack])
                if data_valid_x is None:
                    data_valid_x = sampled.to_numpy()
                else:
                    data_valid_x = np.concatenate((data_valid_x, sampled.to_numpy()))

                if data_valid_y is None:
                    data_valid_y = np.array([label_dict[attack]] * validation_attacks[attack])
                else:
                    data_valid_y = np.concatenate(
                        (data_valid_y, np.array([label_dict[attack]] * validation_attacks[attack])))
                df = pd.concat([df, sampled]).drop_duplicates(keep=False)
                data_frames[device][attack] = df
            data_valid_y = data_valid_y.reshape((len(data_valid_y), 1))
            validation_sets.append((data_valid_x, data_valid_y))

        # pick and sample train sets
        for device, attacks, _ in train_devices:
            data_x, data_y = None, None
            for attack in attacks:
                df = data_frames[device][attack]
                train_requested = total_data_request_count_train[device.value + "-" + attack.value]
                valid_requested = total_data_request_count_valid[device.value + "-" + attack.value]
                test_requested = total_data_request_count_test[device.value + "-" + attack.value]
                total_available_for_train = total_data_available_count[
                                      device.value + "-" + attack.value] - test_requested - valid_requested
                if train_requested > total_available_for_train:
                    # participant's percentage of the remaining training data, ensures data is utilized maximally
                    n_to_pick = floor(float(total_available_for_train) * attacks[attack] / train_requested)
                    picked = df.sample(n=n_to_pick)
                    df = pd.concat([df, picked]).drop_duplicates(keep=False)
                    picked = picked.sample(n=attacks[attack], replace=True)

                    if data_x is None:
                        data_x = picked.to_numpy()
                    else:
                        data_x = np.concatenate((data_x, picked.to_numpy()))
                else:
                    sampled = df.sample(n=attacks[attack])
                    if data_x is None:
                        data_x = sampled.to_numpy()
                    else:
                        data_x = np.concatenate((data_x, sampled.to_numpy()))

                    df = pd.concat([df, sampled]).drop_duplicates(keep=False)

                if data_y is None:
                    data_y = np.array([label_dict[attack]] * attacks[attack])
                else:
                    data_y = np.concatenate((data_y, np.array([label_dict[attack]] * attacks[attack])))

                data_frames[device][attack] = df
            data_y = data_y.reshape((len(data_y), 1))
            train_sets.append((data_x, data_y))
        # NOTE: We average the man and min value for stability reasons!
        scalers = [MinMaxScaler(clip=True).fit(x[0]) for x in train_sets]
        scaler = MinMaxScaler(clip=True)  #
        scaler.min_ = np.stack([s.min_ for s in scalers], axis=1).mean(axis=1)
        scaler.scale_ = np.stack([s.scale_ for s in scalers], axis=1).mean(axis=1)
        return [(scaler.transform(x), y, scaler.transform(validation_sets[idx][0]), validation_sets[idx][1]) for
                idx, (x, y) in enumerate(train_sets)], [(scaler.transform(x), y) for x, y in test_sets]
