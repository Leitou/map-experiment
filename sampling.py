from collections import defaultdict
from math import floor
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from custom_types import RaspberryPi, Attack
from utils import read_data

# TODO:
#  remove noisy paths (ras44gb_path dict)
#  adapt for samples from missing two devices
#  -> evtl make global variables defining device types
#  -> build dicts for paths and adapt sampler constructor
#  adapt sampler class to multiclass classification if needed
#  -> use a different read_data function, building different targets


# assuming everything not normal is malicious
malicious = defaultdict(lambda: 1)
malicious["normal"] = 0
malicious["normal_v2"] = 0

class_map_binary: Dict[Attack, int] = defaultdict(lambda: 1)
class_map_binary[Attack.NORMAL] = 0
class_map_binary[Attack.NORMAL_V2] = 0

class_map_multi: Dict[Attack, int] = defaultdict(lambda: 0)
class_map_multi[Attack.DELAY] = 1
class_map_multi[Attack.DISORDER] = 2
class_map_multi[Attack.FREEZE] = 3
class_map_multi[Attack.HOP] = 4
class_map_multi[Attack.HOP] = 4
class_map_multi[Attack.MIMIC] = 5
class_map_multi[Attack.NOISE] = 6
class_map_multi[Attack.REPEAT] = 7
class_map_multi[Attack.SPOOF] = 8

data_file_paths: Dict[RaspberryPi, Dict[Attack, str]] = {
    RaspberryPi.PI3_2GB: {
        Attack.NORMAL: "data/ras-3-data/samples_normal_2021-06-18-15-59_50s",
        Attack.NORMAL_V2: "data/ras-3-data/samples_normal_v2_2021-06-23-16-54_50s",
        Attack.DELAY: "data/ras-3-data/samples_delay_2021-07-01-08-30_50s",
        Attack.DISORDER: "data/ras-3-data/samples_disorder_2021-06-30-23-54_50s",
        Attack.FREEZE: "data/ras-3-data/samples_freeze_2021-07-01-14-11_50s",
        Attack.HOP: "data/ras-3-data/samples_hop_2021-06-29-23-23_50s",
        Attack.MIMIC: "data/ras-3-data/samples_mimic_2021-06-30-10-33_50s",
        Attack.NOISE: "data/ras-3-data/samples_noise_2021-06-30-19-44_50s",
        Attack.REPEAT: "data/ras-3-data/samples_repeat_2021-07-01-20-00_50s",
        Attack.SPOOF: "data/ras-3-data/samples_spoof_2021-06-30-14-49_50s"
    },
    RaspberryPi.PI4_2GB: {
        Attack.NORMAL: "data/ras-4-black/samples_normal_2021-07-11-22-19_50s",
        Attack.DELAY: "data/ras-4-black/samples_delay_2021-06-30-14-03_50s",
        Attack.DISORDER: "data/ras-4-black/samples_disorder_2021-06-30-09-44_50s",
        Attack.HOP: "data/ras-4-black/samples_hop_2021-06-30-18-24_50s"
    },
    RaspberryPi.PI4_4GB: {
        Attack.NORMAL: "data/ras-4-noisy/samples_normal_2021-06-18-16-09_50s",
        Attack.NORMAL_V2: "data/ras-4-noisy/samples_normal_v2_2021-06-23-16-56_50s",
        Attack.DELAY: "data/ras-4-data/samples_delay_2021-07-01-08-36_50s",
        Attack.DISORDER: "data/ras-4-data/samples_disorder_2021-06-30-23-57_50s",
        Attack.FREEZE: "data/ras-4-data/samples_freeze_2021-07-01-14-13_50s",
        Attack.HOP: "data/ras-4-data/samples_hop_2021-06-29-23-25_50s",
        Attack.MIMIC: "data/ras-4-data/samples_mimic_2021-06-30-10-00_50s",
        Attack.NOISE: "data/ras-4-data/samples_noise_2021-06-30-19-48_50s",
        Attack.REPEAT: "data/ras-4-data/samples_repeat_2021-07-01-20-06_50s",
        Attack.SPOOF: "data/ras-4-data/samples_spoof_2021-06-30-14-54_50s"
    },
}

ras3_paths = {
    "normal": "data/ras-3-data/samples_normal_2021-06-18-15-59_50s",
    "normal_v2": "data/ras-3-data/samples_normal_v2_2021-06-23-16-54_50s",
    "delay": "data/ras-3-data/samples_delay_2021-07-01-08-30_50s",
    "disorder": "data/ras-3-data/samples_disorder_2021-06-30-23-54_50s",
    "freeze": "data/ras-3-data/samples_freeze_2021-07-01-14-11_50s",
    "hop": "data/ras-3-data/samples_hop_2021-06-29-23-23_50s",
    "mimic": "data/ras-3-data/samples_mimic_2021-06-30-10-33_50s",
    "noise": "data/ras-3-data/samples_noise_2021-06-30-19-44_50s",
    "repeat": "data/ras-3-data/samples_repeat_2021-07-01-20-00_50s",
    "spoof": "data/ras-3-data/samples_spoof_2021-06-30-14-49_50s"
}

ras44gb_paths = {
    "normal": "data/ras-4-noisy/samples_normal_2021-06-18-16-09_50s",
    "normal_v2": "data/ras-4-noisy/samples_normal_v2_2021-06-23-16-56_50s",
    "delay": "data/ras-4-data/samples_delay_2021-07-01-08-36_50s",
    "disorder": "data/ras-4-data/samples_disorder_2021-06-30-23-57_50s",
    "freeze": "data/ras-4-data/samples_freeze_2021-07-01-14-13_50s",
    "hop": "data/ras-4-data/samples_hop_2021-06-29-23-25_50s",
    "mimic": "data/ras-4-data/samples_mimic_2021-06-30-10-00_50s",
    "noise": "data/ras-4-data/samples_noise_2021-06-30-19-48_50s",
    "repeat": "data/ras-4-data/samples_repeat_2021-07-01-20-06_50s",
    "spoof": "data/ras-4-data/samples_spoof_2021-06-30-14-54_50s"
}

ras42gb_paths = {
    # to be filled in
}


class DataSampler:
    def __init__(self, sample_size, monitoring_programs):
        assert len(monitoring_programs) >= 1, "At least one monitoring program must be chosen"
        self.num_samples = sample_size
        self.monitoring_paths = []
        self.monitoring_programs = []
        # extract of form [(dt,[progs]),()..]
        for device_type, progs in monitoring_programs:
            assert device_type == "ras4-4gb" or device_type == "ras3", "Device type must be either 3 or 4"
            self.monitoring_programs.extend(progs)
            if device_type == "ras3":
                self.monitoring_paths.extend([ras3_paths[p] for p in progs])

            elif device_type == "ras4-4gb":
                self.monitoring_paths.extend([ras44gb_paths[p] for p in progs])
            else:
                pass

        self.data = None
        self.targets = None

    def get_num_malicious(self):
        num_norm, num_mal = 0.0, 0.0
        mal_progs = []
        for p in self.monitoring_programs:
            if "normal" in p:
                num_norm += 1
                mal_progs.append(0)
            else:
                num_mal += 1
                mal_progs.append(1)

        if num_mal >= 1 and num_norm >= 1:
            num_mal_samples = int(self.num_samples / 2)
            num_norm_samples = num_mal_samples
        elif num_mal == 0 and num_norm >= 1:
            num_mal_samples = 0
            num_norm_samples = self.num_samples
        else:
            num_mal_samples = self.num_samples
            num_norm_samples = 0

        print(f"Sampling: {num_norm_samples} normal samples, {num_mal_samples} malicious samples")
        return mal_progs, num_mal_samples, num_norm_samples

    def get_all_data(self):
        all_data, all_targets = [], []
        for i, pr in enumerate(self.monitoring_programs):
            d, t = read_data(self.monitoring_paths[i], malicious[pr])
            all_data.append(d)
            all_targets.append(t)
        return all_data, all_targets

    def get_num_samples_per_program(self, mal_progs, num_mal_samples, num_norm_samples):
        num_samples_per_prog = []
        for p in mal_progs:
            if p == 1:
                num_prog_samples = int(num_mal_samples / sum(mal_progs))
                num_samples_per_prog.append(num_prog_samples)
            else:
                num_prog_samples = int(num_norm_samples / (len(mal_progs) - sum(mal_progs)))
                num_samples_per_prog.append(num_prog_samples)
        print(f"samples per prog: {num_samples_per_prog}")
        return num_samples_per_prog

    def sample_first_prog(self, first_data, first_targets, num_first_samples):
        prlen = len(first_data)
        self.data, self.targets = first_data, first_targets
        # down sampling for the first program
        if prlen > num_first_samples:
            remaining_idx = np.random.choice(prlen, num_first_samples)
            self.data = self.data[remaining_idx]
            self.targets = self.targets[remaining_idx]

        # up sampling for the first program
        if prlen < num_first_samples:
            num_missing = num_first_samples - prlen
            missing_idxs = np.random.choice(prlen, num_missing)
            self.data = np.vstack((self.data, self.data[missing_idxs]))
            self.targets = np.concatenate((self.targets, self.targets[missing_idxs]))
        print(f"sample {self.monitoring_programs[0]}: data len: {len(self.data)}, targets len: {len(self.targets)}")

    def sample_remaining_progs(self, num_samples_per_program, data_per_prog, targets_per_prog):
        for pr, num_prsamples, prdata, prtargets in zip(self.monitoring_programs[1:], num_samples_per_program[1:],
                                                        data_per_prog[1:], targets_per_prog[1:]):
            prlen = len(prdata)
            # down sampling
            if prlen > num_prsamples:
                remaining_idx = np.random.choice(prlen, num_prsamples)
                self.data = np.vstack((self.data, prdata[remaining_idx]))
                self.targets = np.concatenate((self.targets, prtargets[remaining_idx]))

            # up sampling
            if prlen < num_prsamples:
                multi_idxs = np.random.choice(prlen, num_prsamples)
                self.data = np.vstack((self.data, prdata[multi_idxs]))
                self.targets = np.concatenate((self.targets, prtargets[multi_idxs]))

            print(f"adding {pr} samples: data len: {len(self.data)}, targets len: {len(self.targets)}")
        print()

    # assumes that there is at least one monitoring program
    def sample(self):
        '''read all data and place in a list, perform up or downsampling for each monitoring program,
        such that the participant contributes with num_samples to the federation in a fashion where the number of normal
        samples and number of malicious samples is balanced'''
        print("participant sampling start")

        # find how many samples need to be drawn per program
        mal_progs, num_mal_samples, num_norm_samples = self.get_num_malicious()
        num_samples_per_program = self.get_num_samples_per_program(mal_progs, num_mal_samples, num_norm_samples)

        # populate self.data/targets with corresponding nr of samples of first program
        data_per_prog, targets_per_prog = self.get_all_data()
        self.sample_first_prog(data_per_prog[0], targets_per_prog[0], num_samples_per_program[0])

        # stack the data of the remaining programs to self.data/targets
        self.sample_remaining_progs(num_samples_per_program, data_per_prog, targets_per_prog)

        assert len(self.data) == sum(num_samples_per_program) and \
               len(self.targets) == sum(num_samples_per_program), \
            "Up/Downsampling Failure"

        return self.data, self.targets

    @staticmethod
    def get_all_clients_train_data_and_scaler(train_devices: List[Tuple[RaspberryPi, Dict[Attack, int]]],
                                              test_devices: List[Tuple[RaspberryPi, Dict[Attack, int]]],
                                              multi_class=False) -> \
            Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:

        assert len(train_devices) > 0 and len(
            test_devices) > 0, "Need to provide at least one train and one test device!"
        # Dictionaries that hold total request: e. g. we want 500 train data for a pi3 and delay
        # but may only have 100 -> oversample and prevent overlaps
        total_data_request_count_train: Dict[str, int] = defaultdict(lambda: 0)
        total_data_request_count_test: Dict[str, int] = defaultdict(lambda: 0)
        total_data_available_count: Dict[str, int] = defaultdict(lambda: 0)

        # determine how to label data
        label_dict = class_map_multi if multi_class else class_map_binary

        # pandas data frames for devices and attacks that are requested
        data_frames: Dict[RaspberryPi, Dict[Attack, pd.DataFrame]] = {}

        for device, attacks in train_devices:
            if device not in data_frames:
                data_frames[device] = {}
            for attack in attacks:
                total_data_request_count_train[device.value + "-" + attack.value] += attacks[attack]
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
            if total_data_request_count_test[key] > total_data_available_count[key]:
                raise ValueError(
                    f'Too much data requested for {key}. Please lower sample number! '
                    f'Available: {total_data_available_count[key]}, '
                    f'but requested {total_data_request_count_test[key]}')

        train_sets = []
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

        # pick and sample train sets
        for device, attacks in train_devices:
            data_x, data_y = None, None
            for attack in attacks:
                df = data_frames[device][attack]
                train_requested = total_data_request_count_train[device.value + "-" + attack.value]
                test_requested = total_data_request_count_test[device.value + "-" + attack.value]
                total_requested = train_requested + test_requested
                total_available = total_data_available_count[device.value + "-" + attack.value] - test_requested
                if total_requested > total_available:
                    n_to_pick = floor(attacks[attack] * float(total_available) / total_requested)
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

        # TODO: we need to stabilize the minmaxscaler: either remove extreme outliers
        # or aggregate the standard scalers (e. g. by taking mean average, stddev)
        scaler = StandardScaler() # MinMaxScaler(clip=True)
        all_train_x = np.concatenate(tuple([x[0] for x in train_sets]))
        scaler.fit(all_train_x)
        return [(scaler.transform(x), y) for x, y in train_sets], [(scaler.transform(x), y) for x, y in test_sets]
