import numpy as np
# import matplotlib.pyplot as plt
import time
from random import normalvariate
from multiprocessing import Process, Manager
import json
import os
os.chdir('../')

json_data_struct = {'traffic': []}


def get_bit_rate():
    """
    Select a bit rate (Gbps) for a given
    traffic request. The selection follows
    the Normal distribution.
    :return: selected bit rate
    """
    bit_rates_Gbps = [50, 100, 150, 200]
    position_range = np.arange(0, 100)
    mu = np.mean(position_range)
    sigma = np.std(position_range)
    index = int(normalvariate(mu, sigma) + 0.5)
    if index <= 25:
        pt = 0
    elif 26 <= index <= 50:
        pt = 1
    elif 51 <= index <= 75:
        pt = 2
    else:
        pt = 3
    return bit_rates_Gbps[pt]


def get_time_duration():
    """
    Select a time duration (seconds) for a given
    traffic request. The selection follows
    the Normal distribution.
    :return: selected duration time
    """
    times = np.arange(10, 110, 10)
    position_range = np.arange(0, 100)
    mu = np.mean(position_range)
    sigma = np.std(position_range)
    index = int(normalvariate(mu, sigma) + 0.5)
    if index <= 10:
        pt = 0
    elif 11 <= index <= 20:
        pt = 1
    elif 21 <= index <= 30:
        pt = 2
    elif 31 <= index <= 40:
        pt = 3
    elif 41 <= index <= 50:
        pt = 4
    elif 51 <= index <= 60:
        pt = 5
    elif 61 <= index <= 70:
        pt = 6
    elif 71 <= index <= 80:
        pt = 7
    elif 81 <= index <= 90:
        pt = 8
    else:
        pt = 9
    return times[pt]


def append_to_json_file(file_name, tr, br, load, selected, arrv_t):
    """
    Append data to JSON structure
    :param file_name: json file name
    :param tr: traffic ID
    :param br: bit rate
    :param load: current load in network
    :param selected: selected (if any)
    :param arrv_t: arrival time of request
    :return:
    """
    json_data_struct['traffic'].append({
        'id': tr,
        'bit_rate': br,
        'tr_load': load,
        'tr_selected': selected,
        'tr_arrival_time': arrv_t
    })
    with open(file_name, 'w+') as outfile:
        json.dump(json_data_struct, outfile)


class TrafficGenerator:

    def __init__(self, tr_file_name):
        # Enable multi-process wavelengths editing
        manager = Manager()
        self.wavelengths = manager.dict()
        self.off = 'off'

        init_wavelengths_proc = Process(target=self.init_wavelength_dict)
        init_wavelengths_proc.start()
        init_wavelengths_proc.join()

        # 1) Generate multiple traffic request with time
        # intervals between each other following the
        # Poisson distribution
        tr_events_per_second = 1/30.0  # 1 traffic request arriving (average) every 30 seconds
        seconds = 86400  # 10 hours observation time
        tr_events = np.random.choice([0, 1], size=seconds, replace=True,
                                     p=[1-tr_events_per_second, tr_events_per_second])
        tr_occurrence_times = np.where(tr_events == 1)[0]
        tr_waiting_times = np.diff(tr_occurrence_times)

        # Visualize the arrival times
        # y_ax = [1] * len(occurrence_times)
        # frame1 = plt.gca()
        # plt.scatter(occurrence_times, y_ax)
        # plt.title("%s traffic requests at arriving times" % len(occurrence_times))
        # plt.xlabel("Arriving time (seconds)")
        # frame1.axes.get_yaxis().set_ticks([])
        # plt.show()

        # 2) Iterate through the generated traffic requests
        # arrival times and create tr-instances
        # characteristics in terms of bit rate and lifetime.
        # And create domain json files with them.
        # - considering 90-wavelength available systems
        # - considering 25GBaud with PM-BPSK for 50 Gbps per channel
        tr_prev_arriv_time = 0
        tr_id = 1
        procs = []
        start_time = time.time()

        for tr_arriv_time in tr_occurrence_times[0:10]:
            tr_wait_time = (tr_arriv_time - tr_prev_arriv_time)/10000.0
            tr_prev_arriv_time = tr_arriv_time
            # First wait (i.e., sleep) and then process
            time.sleep(tr_wait_time)

            tr_bit_rate = get_bit_rate()
            tr_time_duration = get_time_duration()/10000.0
            serializable_wavelengths = self.wavelengths.copy()
            tr_selected_load = self.get_selected_load(tr_bit_rate)
            append_to_json_file(tr_file_name, tr_id, tr_bit_rate,
                                serializable_wavelengths, tr_selected_load, int(tr_arriv_time))

            p = Process(target=self.release_load, args=(tr_time_duration, tr_selected_load))
            p.start()
            procs.append(p)

            tr_id += 1

        for p in procs:
            p.join()
        end_time = time.time() - start_time

        print("Took Proc. %s - %s sec to process 10 TRs" % (os.getpid(), end_time))

    def init_wavelength_dict(self):
        static_wavelengths = {k: self.off for k in range(1, 10)}
        self.wavelengths.update(static_wavelengths)

    def available_wavelengths(self):
        """
        Check for wavelengths 'off'
        :return: wavelength indexes where 'off'
        """
        return [key for key, value in self.wavelengths.items() if value == self.off]

    def get_selected_load(self, bit_rate):
        """
        Calculate and select the number of wavelengths
        required for a traffic request to be fulfilled.
        Update wavelength global dict
        :param bit_rate: bit rate of traffic request
        :return: selected wavelengths in domain
        """
        transmission_per_channel = 25  # Gbps
        avail_wavelengths = self.available_wavelengths()
        required_load = int(bit_rate / transmission_per_channel)
        if len(avail_wavelengths) < required_load:
            return None
        selected_load = avail_wavelengths[:required_load - 1]
        for k in selected_load:
            self.wavelengths[k] = 'on'
        return avail_wavelengths[:required_load - 1]

    def release_load(self, sleep_time, selected_load):
        """
        Release wavelength load from the
        wavelength global dict
        :param sleep_time: sleep time in seconds
        :param selected_load: wavelengths to be released
        :return:
        """
        if selected_load:
            time.sleep(sleep_time)
            if selected_load:
                for k in selected_load:
                    self.wavelengths[k] = 'off'
