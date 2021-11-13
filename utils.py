from typing import Tuple, Any, List, Dict, Union

import numpy as np
from math import floor
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from custom_types import RaspberryPi, Behavior


# TODO: add some Reporting Utils. Like
#   a) print accuracies as table
#   b) show accuracies as heatmap
#   c) show thresholds (as heatmap or table, tbd)
#       could then remove the print_experiment_scores thing
class FederationUtils:
    @staticmethod
    def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, Any]:
        correct = np.count_nonzero(y_test == y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=1)
        cm_fed = confusion_matrix(y_test, y_pred)  # could also extract via tn, fp, fn, tp = confusion_matrix().ravel()
        return correct / len(y_pred), f1, cm_fed

    @staticmethod
    def get_confusion_matrix_vals_in_percent(acc, conf_mat, behavior):
        if acc == 1.0:
            if behavior in [Behavior.NORMAL, Behavior.NORMAL_V2]:
                tn, fp, fn, tp = 1, 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, 1
        else:
            tn, fp, fn, tp = conf_mat.ravel() / sum(conf_mat.ravel())
        return tn, fp, fn, tp

    @staticmethod
    def print_experiment_scores(y_test: np.ndarray, y_pred: np.ndarray, federated=True):
        if federated:
            print("\n\nResults Federated Model:")
        else:
            print("\n\nResults Centralized Model:")

        accuracy, f1, cm_fed = FederationUtils.calculate_metrics(y_test, y_pred)
        print(classification_report(y_test, y_pred, target_names=["Normal", "Infected"])
              if len(np.unique(y_pred)) > 1
              else "only single class predicted, no report generated")
        print(f"Details:\nConfusion matrix \n[(TN, FP),\n(FN, TP)]:\n{cm_fed}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%, F1 score: {f1 * 100:.2f}%")

    @staticmethod
    def plot_f1_scores():
        fig, ax = plt.subplots(figsize=(12, 5))
        #     ax.set_title('Average, min and max F1-Scores under the ' +  attack_str + ' attack')
        ax.set_ylabel('F1-Score (%)')
        ax.set_ylim(0, 100.)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        #     plt.gca().spines['bottom'].set_visible(False)

        pos_x = 0
        colors = ['limegreen', 'gold', 'tab:orange', 'orangered']
        colors = ['#87d64b', '#fae243', '#f8961e', '#ff4d36']
        text_colors = ['#456e25', '#998a28', '#b06a13', '#b33424']

        aggs = ['avg', 'med', 'tm1', 'tm2', 'tm2-2rs']

        bar_width = 1.5
        sep = 1

        for agg in aggs:
            for f in [0, 1, 2, 3]:
                color = colors[f]
                text_color = text_colors[f]
                if f == 0:
                    results = get_all_results(
                        'test_results/decentralized_classifier_fedsgd/NO_ATTACK/0,95 5rr 64bs ' + agg + '/')
                else:
                    results = get_all_results(base_path + agg + ' vs ' + repr(f) + end_path)
                mean_f1 = get_mean_score(results) * 100
                if mean_f1 == 0.0:
                    mean_f1_bar = 1
                else:
                    mean_f1_bar = mean_f1
                min_f1 = get_min_score(results) * 100
                max_f1 = get_max_score(results) * 100

                yerr_up = max_f1 - mean_f1
                yerr_down = mean_f1 - min_f1

                if agg == 'avg':
                    ax.bar(pos_x, height=mean_f1_bar, color=color, width=bar_width, lw=0.7, edgecolor='black',
                           label='f=' + repr(f))
                else:
                    ax.bar(pos_x, height=mean_f1_bar, color=color, width=bar_width, lw=0.7,
                           edgecolor='black')  # yerr=[[yerr_down], [yerr_up]], capsize=11

                ax.errorbar(x=pos_x, y=mean_f1_bar, yerr=[[yerr_down], [yerr_up]],
                            capsize=6, color='black', elinewidth=0, lw=1.0, solid_capstyle='round')

                s = percentage_to_text(mean_f1)
                if len(s) == 1:
                    text_x = pos_x - 0.2
                else:
                    text_x = pos_x - 0.5

                ax.text(x=text_x, y=mean_f1_bar + 1.2, s=s, fontsize='15.5', color=text_color)

                pos_x += bar_width
            pos_x += sep

        ticks = [(1.5 * bar_width) + (4 * bar_width + sep) * i for i in range(0, 5)]

        ax.set_xticks(ticks)
        aggs = [agg.upper() for agg in aggs]
        ax.set_xticklabels(aggs)
        ax.legend(bbox_to_anchor=(1.12, 0.5), loc='right')
        plt.show()
        fig.savefig('f1_scores_' + '_'.join(attack_str.split(' ')) + '.pdf', bbox_inches='tight')


# Assumption we test at most on what we train (attack types)
def select_federation_composition(participants_per_arch: List, normals: List[Tuple[Behavior, int]],
                                  attacks: List[Behavior],
                                  val_percentage: float, attack_frac: float,
                                  num_behavior_test_samples: int, is_anomaly: bool = False) \
        -> Tuple[List[Tuple[Any, Dict[Behavior, Union[int, float]], Dict[Behavior, Union[int, float]]]], List[
            Tuple[Any, Dict[Behavior, int]]]]:
    assert len(list(RaspberryPi)) == len(participants_per_arch), "lengths must be equal"
    assert normals[0][1] == normals[1][1] if len(
        normals) == 2 else True, "equal amount of normal version samples required"
    # populate train and test_devices for
    train_devices, test_devices = [], []
    for i, num_p in enumerate(participants_per_arch):
        for p in range(num_p):

            # add all normal monitorings for the training + validation + testing per participant
            train_d, val_d = {}, {}
            for normal in normals:
                train_d[normal[0]] = normal[1]
                val_d[normal[0]] = floor(normal[1] * val_percentage)

            # add all attacks for training + validation per participant in case of binary classification training
            if not is_anomaly:
                for attack in attacks:
                    train_d[attack] = floor(normals[0][1] * attack_frac)
                    val_d[attack] = floor(normals[0][1] * attack_frac * val_percentage)
            train_devices.append((list(RaspberryPi)[i], train_d, val_d))

            # populate test dictionary with all behaviors (only once per device type)
            if p == 0:
                for b in list(Behavior):
                    test_d = {}
                    test_d[b] = num_behavior_test_samples
                    test_devices.append((list(RaspberryPi)[i], test_d))

    return train_devices, test_devices


# helper function independent of how test or train_devices are created
# can be used to plot exactly how many samples of each device are being used for training to estimate the oversampling
def get_sampling_per_device(train_devices, test_devices, include_train=True, incl_val=True, include_test=False):
    devices_sample_reqs = []  # header
    for d in RaspberryPi:
        device_samples = [d.value]
        for b in Behavior:
            bcount = 0
            for dev, train_d, val_d in train_devices:
                if dev == d:
                    if include_train:
                        if b in train_d:
                            bcount += train_d[b]
                    if incl_val:
                        if b in val_d:
                            bcount += val_d[b]
            if include_test:
                for dev, test_d in test_devices:
                    if dev == d:
                        if b in test_d:
                            bcount += test_d[b]
            device_samples.append(bcount)
        normals = sum(device_samples[1:3])
        attacks = sum(device_samples[3:])
        device_samples.append(normals / attacks if attacks != 0 else None)
        devices_sample_reqs.append(device_samples)
    return devices_sample_reqs
