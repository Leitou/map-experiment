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
    def __init__(self, device_type, monitoring_programs):
        assert device_type == 4 or device_type == 3, "Device type must be either 3 or 4"
        assert len(monitoring_programs) >= 1, "At least one monitoring program must be chosen"
        self.monitoring_programs = monitoring_programs
        if device_type == 3:
            self.monitoring_paths = [ras3_paths[p] for p in monitoring_programs]
        else:
            self.monitoring_paths = [ras4_paths[p] for p in monitoring_programs]
        self.data = None
        self.targets = None


    # TODO: How to include the StandardScaler? Not possible to standardize on the whole global data
    # assumes that there is at least one monitoring program
    def sample(self, num_samples):
        '''read all data and place in a list, perform up or downsampling for each monitoring program,
        such that the participant contributes with num_samples to the federation'''

        # TODO: make balanced for num_normal/num_malicious (Rule: 10 * num_infeatures ~ 800 samples at min.)
        # check whether monitoring programs contain "normal" or not
        # cases:
        # only malicious
        # only healthy
        # both normal and malicious once
        # both normal and malicious at least once
        # to be balanced:
        # -> num normal & normal_v2 samples == num malicious samples

        # case mixed: most important
        # num_normal, num_malicious
        # condition: num_samples / 2 == num_normal and num_samples / 2 = num_malicious
        # -> sample as usual in the loop by stacking/concatenating but calculating
        # num_samples_per_program dependent on num_normal or num_malicious respectively

        all_data, all_targets = [], []
        for i, pr in enumerate(self.monitoring_programs):
            d, t = read_data(self.monitoring_paths[i], malicious[pr])
            all_data.append(d)
            all_targets.append(t)

        num_samples_per_program = int(num_samples / len(self.monitoring_programs))

        prlen = len(all_targets[0])
        self.data, self.targets = all_data[0], all_targets[0]
        # down sampling for the first program
        if prlen > num_samples_per_program:
            remaining_idx = np.random.choice(prlen, num_samples_per_program)
            self.data = self.data[remaining_idx]
            self.targets = self.targets[remaining_idx]

        # up sampling for the first program
        if prlen < num_samples_per_program:
            num_missing = num_samples_per_program - prlen
            missing_idxs = np.random.choice(prlen, num_missing)
            self.data = np.vstack((self.data, self.data[missing_idxs]))
            self.targets = np.concatenate((self.targets, self.targets[missing_idxs]))

        print(f"data len: {len(self.data)}, targets len: {len(self.targets)}")
        # add the remaining programs to self.data/targets
        i = 1
        for prdata in all_data[1:]:
            prlen = len(all_targets[i])
            # down sampling
            if prlen > num_samples_per_program:
                remaining_idx = np.random.choice(prlen, num_samples_per_program)
                self.data = np.vstack((self.data, prdata[remaining_idx]))
                self.targets = np.concatenate((self.targets, all_targets[i][remaining_idx]))

            # up sampling
            if prlen < num_samples_per_program:
                multi_idxs = np.random.choice(prlen, num_samples_per_program)
                self.data = np.vstack((self.data, prdata[multi_idxs]))
                self.targets = np.concatenate((self.targets, all_targets[i][multi_idxs]))

            print(f"{i}data len: {len(self.data)}, targets len: {len(self.targets)}")
            i += 1


        assert len(self.data) == num_samples_per_program*len(self.monitoring_programs) and \
               len(self.targets) == num_samples_per_program*len(self.monitoring_programs), "Up/Downsampling Failure"

        return self.data, self.targets


# p = ParticipantSampler(3, ["normal"], 100)
# print(p.sample()[0])
# print(p.sample()[1])