import os

from data_handler import DataHandler

if __name__ == "__main__":
    os.chdir("..")
    print("------------------Raw Data Availability------------------")
    DataHandler.show_data_availability(raw=True)
    print("----------------Filtered Data Availability---------------")
    DataHandler.show_data_availability(raw=False)
    print("------------------Constant Columns-----------------------")
    df = DataHandler.parse_all_files_to_df(filter_suspected_external_events=False, filter_constant_columns=False,
                                           filter_outliers=False)
    print(df.columns[df.nunique() <= 1])
