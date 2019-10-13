import multiprocessing
import traffic_generator
import os


def call_traffic_generator(file_name):
    print("*** Starting proc %s..." % os.getpid())
    traffic_generator.TrafficGenerator(file_name)


tr_json_files = ['traffic-files/traffic_domain_1.json',
                 'traffic-files/traffic_domain_2.json',
                 'traffic-files/traffic_domain_3.json',
                 'traffic-files/traffic_domain_4.json',
                 'traffic-files/traffic_domain_5.json']

procs = []
for tr_file_name in tr_json_files:
    p = multiprocessing.Process(target=call_traffic_generator, args=(tr_file_name,))
    p.start()
    procs.append(p)

for p in procs:
    p.join()
