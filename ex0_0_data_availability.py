from data_handler import DataHandler

if __name__ == "__main__":
    DataHandler.show_data_availability(raw=True)
    print("---------------------------------------------------------")
    DataHandler.show_data_availability(raw=False)
