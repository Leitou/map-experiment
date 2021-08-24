import matplotlib.pyplot as plt
import pandas as pd

from custom_types import RaspberryPi, Behavior
from data_handler import DataHandler

if __name__ == "__main__":
    all_data: pd.DataFrame = DataHandler._DataHandler__parse_all_files_to_df()
    col_names = [col for col in all_data if col not in ['device', 'attack']]
    col_names = col_names

    for device in RaspberryPi:
        for normal in [Behavior.NORMAL, Behavior.NORMAL_V2]:
            for attack in [Behavior.FREEZE, Behavior.REPEAT]:
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
                    axs[i].plot(xes_normal, ys_normal, color='green')
                    axs[i].plot(xes_normal, ys_attack_repeated, color='red')
                    axs[i].set_title(col_names[i], fontsize='xx-large')

                fig.savefig(f'data_plot_{device.value}_{normal.value}_{attack.value}.png', figure=fig, dpi=50)
