from utils import read_data
from collections import defaultdict
import numpy as np
from sys import exit

# TODO:
#  remove noisy paths,  standardize filenames to "normal", "delay"
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
# TODO: use clean samples
ras4_paths = {
    "normal": "data/ras-4-noisy/samples_normal_2021-06-18-16-09_50s",
    "delay": "data/ras-4-data/samples_delay_2021-07-01-08-36_50s"
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

    # return the inputs and targets by using read_data customizable
    # TODO: currently sample only works for one monitoring program
    #  make customizable for multiple types of monitorings
    #  -> each monitoring dataset sliced/multiplied to be of the same size
    #  e.g. 1000 samples, three monitorings -> each has 333 samples
    #
    # TODO: How to include the StandardScaler? Not possible to standardize on the whole global data
    def sample(self, num_samples):
        # read all data and place in a list, if the elements are of different lengths perform up or downsampling
        self.data, self.targets = read_data(self.monitoring_paths[0], malicious[self.monitoring_programs[0]])

        l = len(self.targets)
        # down sampling
        if l > num_samples:
            remaining_idx = np.random.choice(l, num_samples)
            self.data = self.data[remaining_idx]
            self.targets = self.targets[remaining_idx]

        # up sampling
        if l < num_samples:
            num_missing = num_samples - l
            multi_idxs = np.random.choice(l, num_missing)
            self.data = np.vstack((self.data, self.data[multi_idxs]))
            self.targets = np.concatenate((self.targets, self.targets[multi_idxs]))

        assert len(self.data) == num_samples and len(self.targets) == num_samples, "Up/Downsampling Failure"

        return self.data, self.targets


# p = ParticipantSampler(3, ["normal"], 100)
# print(p.sample()[0])
# print(p.sample()[1])