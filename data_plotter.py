from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from custom_types import RaspberryPi, Behavior
from data_handler import DataHandler


# TODO: make one function: plot(List[Tuple[Device, Behavior, str]]) to plot list of (Device, Behavior, Color in plot)
# kept for convenience
def plot_one_normal_against_attacks(all_data, col_names, attacks=[Behavior.FREEZE, Behavior.REPEAT]):
    for device in RaspberryPi:
        for normal in [Behavior.NORMAL, Behavior.NORMAL_V2]:
            for attack in attacks:
                fig, axs = plt.subplots(len(col_names))
                fig.suptitle(f'Plotting {device.value}: {normal.value} vs {attack.value}')
                fig.set_figheight(len(col_names) * 4)
                fig.set_figwidth(50)

                for i in range(len(col_names)):
                    df_normal = all_data.loc[
                        (all_data['attack'] == normal.value) & (all_data['device'] == device.value)]
                    df_att = all_data.loc[(all_data['attack'] == attack.value) & (all_data['device'] == device.value)]
                    xes_normal = [i for i in range(len(df_normal))]
                    ys_normal = df_normal[col_names[i]].tolist()
                    ys_attack_real = df_att[col_names[i]].tolist()
                    ys_attack_repeated = []
                    for j in range(len(ys_normal)):
                        ys_attack_repeated.append(ys_attack_real[(j % len(ys_attack_real))])
                    axs[i].plot(xes_normal, ys_normal, color='green', label="normal")
                    axs[i].plot(xes_normal, ys_attack_repeated, color='red', label=attack.value)
                    axs[i].set_title(col_names[i], fontsize='xx-large')
                    axs[i].legend()

                # fig.savefig(f'data_plot_{device.value}_{normal.value}_{attack.value}.png', figure=fig, dpi=50)
                fig.savefig(f'data_plot_{device.value}_{normal.value}_{attack.value}.png', dpi=50)


def plot_normals_against_attacks_per_device2(all_data, col_names, attacks=[Behavior.FREEZE, Behavior.REPEAT]):
    for device in RaspberryPi:
        for attack in attacks:
            fig, axs = plt.subplots(len(col_names))
            fig.suptitle(f'Plotting {device.value}: normals vs {attack.value}')
            fig.set_figheight(len(col_names) * 4)
            fig.set_figwidth(50)

            for i in range(len(col_names)):
                df_normal = all_data.loc[
                    (all_data['attack'] == Behavior.NORMAL.value) & (all_data['device'] == device.value)]
                df_normal_v2 = all_data.loc[
                    (all_data['attack'] == Behavior.NORMAL_V2.value) & (all_data['device'] == device.value)]
                df_att = all_data.loc[(all_data['attack'] == attack.value) & (all_data['device'] == device.value)]

                xes_normal = [i for i in range(len(df_normal))]
                xes_normal_v2 = [i for i in range(len(df_normal_v2))]
                ys_normal = df_normal[col_names[i]].tolist()
                ys_normal_v2 = df_normal_v2[col_names[i]].tolist()
                ys_attack_real = df_att[col_names[i]].tolist()
                ys_attack_repeated = []
                for j in range(len(ys_normal)):
                    ys_attack_repeated.append(ys_attack_real[(j % len(ys_attack_real))])
                axs[i].plot(xes_normal, ys_normal, color='green', label="normal")
                axs[i].plot(xes_normal_v2, ys_normal_v2, color='blue', label="normal_v2")
                axs[i].plot(xes_normal, ys_attack_repeated, color='red', label=attack.value)
                axs[i].set_title(col_names[i], fontsize='xx-large')
                axs[i].legend()

            fig.savefig(f'data_plot_{device.value}_NORMALS_{attack.value}.png', dpi=50)


def plot_behaviors_for_all_devices(all_data, col_names, behaviors, devices=RaspberryPi):
    colors = ["red", "green", "blue", "orange"][:len(devices)]
    for b in behaviors:
        fig, axs = plt.subplots(len(col_names))
        fig.suptitle(f'Plotting {b.value}: all devices')
        fig.set_figheight(len(col_names) * 4)
        fig.set_figwidth(50)
        for i in range(len(col_names)):
            for j, device in enumerate(devices):
                df_b = all_data.loc[(all_data['attack'] == b.value) & (all_data['device'] == device.value)]
                xes_b = [i for i in range(len(df_b))]
                ys_b = df_b[col_names[i]].tolist()
                axs[i].plot(xes_b, ys_b, color=colors[j], label=device.value)
            axs[i].set_title(col_names[i], fontsize='xx-large')
            axs[i].legend()

        fig.savefig(f'data_plot_all_devices_{b.value}.png', dpi=50)


class DataPlotter:
    @staticmethod
    def plot_behaviors(behaviors: List[Tuple[RaspberryPi, Behavior, str]], plot_name: Union[str, None] = None):
        # first find max number of samples
        all_data_parsed = DataHandler.parse_all_files_to_df(raw=True, save_to_file=False)
        max_number_of_samples = 0
        for behavior in behaviors:
            df_behavior = all_data_parsed.loc[
                (all_data_parsed['attack'] == behavior[1].value) & (all_data_parsed['device'] == behavior[0].value)]
            if len(df_behavior) > max_number_of_samples:
                max_number_of_samples = len(df_behavior)
        cols_to_plot = [col for col in all_data_parsed if col not in ['device', 'attack']]

        fig, axs = plt.subplots(len(cols_to_plot))
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot) * 4)
        fig.set_figwidth(50)
        for i in range(len(cols_to_plot)):
            for device, behavior, line_color in behaviors:
                df_b = all_data_parsed.loc[
                    (all_data_parsed['attack'] == behavior.value) & (all_data_parsed['device'] == device.value)]
                xes_b = [i for i in range(max_number_of_samples)]
                ys_actual_b = df_b[cols_to_plot[i]].tolist()
                ys_upsampled_b = [ys_actual_b[i % len(ys_actual_b)] for i in range(max_number_of_samples)]
                axs[i].plot(xes_b, ys_upsampled_b, color=line_color, label=(device.value + " " + behavior.value))
            axs[i].set_title(cols_to_plot[i], fontsize='xx-large')
            axs[i].legend()

        if plot_name is not None:
            fig.savefig(f'data_plot_{plot_name}.png', dpi=100)


if __name__ == "__main__":
    DataPlotter.plot_behaviors(
        [(RaspberryPi.PI4_2GB_WC, Behavior.HOP, "darkred"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NOISE, "red"),
         (RaspberryPi.PI4_2GB_WC, Behavior.SPOOF, "yellow"),
         (RaspberryPi.PI4_2GB_WC, Behavior.DELAY, "goldenrod"),
         (RaspberryPi.PI4_2GB_WC, Behavior.DISORDER, "cyan"),
         (RaspberryPi.PI4_2GB_WC, Behavior.FREEZE, "black"),
         (RaspberryPi.PI4_2GB_WC, Behavior.REPEAT, "blue"),
         (RaspberryPi.PI4_2GB_WC, Behavior.MIMIC, "fuchsia"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "lightgreen"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL_V2, "darkgreen")], plot_name="all_pi4_2gb")

    DataPlotter.plot_behaviors(
        [(RaspberryPi.PI4_2GB_WC, Behavior.DELAY, "goldenrod"),
         (RaspberryPi.PI4_2GB_WC, Behavior.DISORDER, "cyan"),
         (RaspberryPi.PI4_2GB_WC, Behavior.MIMIC, "fuchsia"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "lightgreen"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL_V2, "darkgreen")], plot_name="writeback_affecting_attacks_pi4_2gb")

    DataPlotter.plot_behaviors(
        [(RaspberryPi.PI4_2GB_WC, Behavior.HOP, "darkred"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NOISE, "red"),
         (RaspberryPi.PI4_2GB_WC, Behavior.SPOOF, "yellow"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "lightgreen"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL_V2, "darkgreen")], plot_name="attacks_with_randomness_pi4_2gb")

    DataPlotter.plot_behaviors(
        [(RaspberryPi.PI4_2GB_WC, Behavior.FREEZE, "black"),
         (RaspberryPi.PI4_2GB_WC, Behavior.REPEAT, "blue"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL, "lightgreen"),
         (RaspberryPi.PI4_2GB_WC, Behavior.NORMAL_V2, "darkgreen")], plot_name="freeze_repeat_pi4_2gb")

