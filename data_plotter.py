from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from custom_types import RaspberryPi, Behavior
from data_handler import DataHandler


class DataPlotter:
    @staticmethod
    def plot_behaviors(behaviors: List[Tuple[RaspberryPi, Behavior, str]], plot_name: Union[str, None] = None):
        # first find max number of samples
        all_data_parsed = DataHandler.parse_all_files_to_df(filter_outliers=False)
        max_number_of_samples = 0
        for behavior in behaviors:
            df_behavior = all_data_parsed.loc[
                (all_data_parsed['attack'] == behavior[1].value) & (all_data_parsed['device'] == behavior[0].value)]
            if len(df_behavior) > max_number_of_samples:
                max_number_of_samples = len(df_behavior)
        cols_to_plot = [col for col in all_data_parsed if col not in ['device', 'attack']]

        fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
        axs = axs.ravel().tolist()
        fig.suptitle(plot_name)
        fig.set_figheight(len(cols_to_plot))
        fig.set_figwidth(50)
        for i in range(len(cols_to_plot)):
            for device, behavior, line_color in behaviors:
                df_b = all_data_parsed.loc[
                    (all_data_parsed['attack'] == behavior.value) & (all_data_parsed['device'] == device.value)]
                xes_b = [i for i in range(max_number_of_samples)]
                ys_actual_b = df_b[cols_to_plot[i]].tolist()
                ys_upsampled_b = [ys_actual_b[i % len(ys_actual_b)] for i in range(max_number_of_samples)]
                axs[i].set_yscale('log')
                axs[i].plot(xes_b, ys_upsampled_b, color=line_color, label=(device.value + " " + behavior.value))
            axs[i].set_title(cols_to_plot[i], fontsize='xx-large')
            axs[i].legend()

        if plot_name is not None:
            fig.savefig(f'data_plot_{plot_name}.png', dpi=100)
            print(f'Saved {plot_name}')

    @staticmethod
    def plot_devices_as_kde():
        for device in RaspberryPi:
            plot_name = f"all_device_{device.value}_hist"
            all_data_parsed = DataHandler.parse_all_files_to_df(filter_outliers=True)
            all_data_parsed = all_data_parsed[all_data_parsed.device == device.value]
            cols_to_plot = [col for col in all_data_parsed if col not in ['device', 'attack']]

            all_data_parsed = all_data_parsed.drop(['device'], axis=1)
            all_data_parsed = all_data_parsed.reset_index()
            fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
            axs = axs.ravel().tolist()
            fig.suptitle(plot_name)
            fig.set_figheight(len(cols_to_plot))
            fig.set_figwidth(50)
            palette = {Behavior.NORMAL.value: "green", Behavior.NORMAL_V2.value: "lightgreen",
                       Behavior.DELAY.value: "yellow", Behavior.DISORDER.value: "orange",
                       Behavior.FREEZE.value: "grey", Behavior.HOP.value: "red",
                       Behavior.MIMIC.value: "violet", Behavior.NOISE.value: "turquoise",
                       Behavior.REPEAT.value: "black", Behavior.SPOOF.value: "darkred"}
            for i in range(len(cols_to_plot)):
                axs[i].set_ylim([1e-4, 2])
                for behav in Behavior:
                    if all_data_parsed[all_data_parsed.attack == behav.value][cols_to_plot[i]].unique().size == 1:
                        axs[i].axvline(all_data_parsed[all_data_parsed.attack == behav.value][cols_to_plot[i]].iloc[0],
                                       ymin=1e-4, ymax=2, color=palette[behav.value])
                sns.kdeplot(data=all_data_parsed, x=cols_to_plot[i], palette=palette, hue="attack",
                            common_norm=False, common_grid=True, ax=axs[i], cut=2,
                            log_scale=(False, True))  # False, True

            if plot_name is not None:
                fig.savefig(f'data_plot_{plot_name}.png', dpi=100)

    @staticmethod
    def plot_behaviors_as_kde():
        for behav in Behavior:
            plot_name = f"all_behavior_{behav.value}_hist"
            all_data_parsed = DataHandler.parse_all_files_to_df(filter_outliers=True)
            all_data_parsed = all_data_parsed[all_data_parsed.attack == behav.value]
            cols_to_plot = [col for col in all_data_parsed if col not in ['device', 'attack']]

            all_data_parsed = all_data_parsed.drop(['attack'], axis=1)
            all_data_parsed = all_data_parsed.reset_index()
            fig, axs = plt.subplots(nrows=ceil(len(cols_to_plot) / 4), ncols=4)
            axs = axs.ravel().tolist()
            fig.suptitle(plot_name)
            fig.set_figheight(len(cols_to_plot))
            fig.set_figwidth(50)
            palette = {RaspberryPi.PI3_1GB.value: "red", RaspberryPi.PI4_2GB_WC.value: "blue",
                       RaspberryPi.PI4_2GB_BC.value: "orange", RaspberryPi.PI4_4GB.value: "green"}
            for i in range(len(cols_to_plot)):
                axs[i].set_ylim([1e-4, 2])
                if all_data_parsed[cols_to_plot[i]].unique().size == 1:
                    continue
                for device in RaspberryPi:
                    if all_data_parsed[all_data_parsed.device == device.value][cols_to_plot[i]].unique().size == 1:
                        axs[i].axvline(all_data_parsed[all_data_parsed.device == device.value][cols_to_plot[i]].iloc[0],
                                       ymin=1e-4, ymax=2, color=palette[device.value])
                sns.kdeplot(data=all_data_parsed, x=cols_to_plot[i], palette=palette, hue="device",
                            common_norm=False, common_grid=True, ax=axs[i], cut=2,
                            log_scale=(False, True))  # False, True

            if plot_name is not None:
                fig.savefig(f'data_plot_{plot_name}.png', dpi=100)
