from utils import read_data
from collections import defaultdict
import numpy as np
from sys import exit

# TODO:
#  remove noisy paths,  standardize filenames to "normal", "delay",
#  adapt for samples from missing two devices
#  adapt sampler class to multiclass classification/autoencoder etc


# assuming everything not normal is malicious
malicious = defaultdict(lambda: 1)
malicious["normal"] = 0
malicious["normal_v2"] = 0

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

# TODO: use clean samples for both normal versions
ras4_paths = {
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

class ParticipantSampler():
    def __init__(self, device_type, sample_size, monitoring_programs):
        assert device_type == 4 or device_type == 3, "Device type must be either 3 or 4"
        assert len(monitoring_programs) >= 1, "At least one monitoring program must be chosen"
        self.monitoring_programs = monitoring_programs
        self.num_samples = sample_size
        if device_type == 3:
            self.monitoring_paths = [ras3_paths[p] for p in monitoring_programs]
        else:
            self.monitoring_paths = [ras4_paths[p] for p in monitoring_programs]
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
        for pr, num_prsamples, prdata, prtargets in zip(self.monitoring_programs[1:],num_samples_per_program[1:],
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

    # TODO: make balanced for num_normal/num_malicious (Rule: 10 * num_infeatures ~ 800 samples at min.)
    # assumes that there is at least one monitoring program
    def sample(self):
        '''read all data and place in a list, perform up or downsampling for each monitoring program,
        such that the participant contributes with num_samples to the federation in a fashion where the number of normal
        samples and number of malicious samples is balanced'''

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


# p = ParticipantSampler(3, ["normal"], 100)
# print(p.sample()[0])
# print(p.sample()[1])