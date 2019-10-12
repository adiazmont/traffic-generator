import numpy as np
import matplotlib.pyplot as plt
import time
from random import normalvariate
from multiprocessing import Process, Manager
manager = Manager()
wavelengths = manager.dict()
off = 'off'


def init_wavelength_dict():
    static_wavelengths = {k: off for k in range(1, 10)}
    wavelengths.update(static_wavelengths)


init_wavelengths_proc = Process(target=init_wavelength_dict)
init_wavelengths_proc.start()
init_wavelengths_proc.join()
print(wavelengths)


def get_bit_rate():
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
    times = range(1, 7)
    position_range = np.arange(0, 100)
    mu = np.mean(position_range)
    sigma = np.std(position_range)
    index = int(normalvariate(mu, sigma) + 0.5)
    if index <= 17:
        pt = 0
    elif 18 <= index <= 35:
        pt = 1
    elif 36 <= index <= 53:
        pt = 2
    elif 54 <= index <= 71:
        pt = 3
    elif 72 <= index <= 89:
        pt = 4
    else:
        pt = 5
    return times[pt]


def available_wavelengths():
    """
    Check for wavelengths 'off'
    :return: wavelength indexes where 'off'
    """
    return [key for key, value in wavelengths.items() if value == off]


def get_selected_load(bit_rate):
    transmission_per_channel = 25  # Gbps
    avail_wavelengths = available_wavelengths()
    print("Get selected load")
    print(avail_wavelengths)
    required_load = int(bit_rate/transmission_per_channel)
    if len(avail_wavelengths) < required_load:
        return None
    selected_load = avail_wavelengths[:required_load-1]
    for k in selected_load:
        wavelengths[k] = 'on'
    return avail_wavelengths[:required_load-1]


def release_load(tr, sleep_time, selected_load):
    if selected_load:
        time.sleep(sleep_time)
        print("TS-%s releasing load %s" % (tr, selected_load))
        if selected_load:
            for k in selected_load:
                wavelengths[k] = 'off'


# 1) Generate multiple traffic request with time
# intervals between each other following the
# Poisson distribution
events_per_second = 1/30.0  # 1 traffic request arriving (average) every 30 seconds
seconds = 86400  # 10 hours observation time
events = np.random.choice([0, 1], size=seconds, replace=True,
                          p=[1-events_per_second, events_per_second])
occurrence_times = np.where(events == 1)[0]
waiting_times = np.diff(occurrence_times)

# Visualize the arrival times
# y_ax = [1] * len(occurrence_times)
# frame1 = plt.gca()
# plt.scatter(occurrence_times, y_ax)
# plt.title("%s traffic requests at arriving times" % len(occurrence_times))
# plt.xlabel("Arriving time (seconds)")
# frame1.axes.get_yaxis().set_ticks([])
# plt.show()


# 2) Iterate through the generated traffic request
# arrival times and create traffic request
# characteristics in terms of bit rate and lifetime
# - considering 90-wavelength available systems
# - considering 25GBaud with PM-BPSK for 50 Gbps per channel
prev_arriv_time = 0
tr_id = 1
procs = []
for arriv_time in occurrence_times[0:5]:
    wait_time = (arriv_time - prev_arriv_time)/100.0
    prev_arriv_time = arriv_time
    # First wait (i.e., sleep) and then process
    time.sleep(wait_time)

    tr_bit_rate = get_bit_rate()
    tr_time_duration = get_time_duration()/1.0
    tr_selected_load = get_selected_load(tr_bit_rate)
    print("TR-%s selecting load %s" % (tr_id, tr_selected_load))

    p = Process(target=release_load, args=(tr_id, tr_time_duration, tr_selected_load))

    p.start()
    procs.append(p)

    tr_id += 1

for p in procs:
    p.join()

