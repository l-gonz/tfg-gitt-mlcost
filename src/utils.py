import platform
import subprocess
import re
import os
import psutil
import math

from codecarbon.output import BaseOutput


FILE_NAME = 'output.csv'

COLUMN_NAMES = ["dataset", "model", "cpu_load", "accuracy", "time", "emissions", "energy_consumed"]

UNITS = ["k", "", "m", "μ", "n"]
DAY = 24*60*60
HOUR = 60*60
MINUTE = 60


class EmissionsOutput(BaseOutput):
    """Output class for codecarbon's emission tracker that stores
    all the data recorded."""
    def out(self, emissions_data):
        self.data = emissions_data


def print_output(name, score, time, emissions, energy):
    """Print model scores to standard output."""
    print(name)
    if isinstance(score, float): 
        print(f'Accuracy: {score:.2%}')
    else:
        print(f"Score:\n{score}")

    days, s = divmod(time, DAY)
    hours, s = divmod(s, HOUR)
    mins, secs = divmod(s, MINUTE)
    time_format = "Time: " + \
        ("{days} days " if days > 0 else "") + \
        ("{hours} hours " if hours > 0 else "") + \
        ("{mins} min " if mins > 0 else "") + \
        ("{secs:.4g} s" if secs > 1 else "") + \
        ("{milis:.4g} ms" if secs < 1 else "")
    print(time_format.format(days=int(days), hours=int(hours), mins=int(mins), secs=secs, milis=secs * 1000))

    exp = math.floor(math.log10(emissions)) // 3
    unit = UNITS[abs(exp)]
    print(f"Emissions: {emissions/ 10**(3*exp):.2f} {unit}g (CO2-equivalents)")

    exp = math.floor(math.log10(energy)) // 3
    unit = UNITS[abs(exp)]
    print(f"Energy consumed: {energy/ 10**(3*exp):.2f} {unit}Wh")
    print("---------------------------")


def print_computer_info():
    """Print platform, CPU and RAM info to standard output."""
    print("---------------------------")
    print("Running on " + platform.node())
    try:
        print(platform.freedesktop_os_release()['PRETTY_NAME'] + " " + platform.machine())
    except AttributeError:
        print(f"{platform.system()} {platform.release()}")
        
    print("Python " + platform.python_version())
    if platform.system() == "Linux":
        output = subprocess.check_output("cat /proc/cpuinfo", shell=True).strip().decode().split('\n')
        cpu_info = {item[0]: item[1] for item in [re.split("\s*:\s*", line, maxsplit=2) for line in output if line]}
        if 'model name' in cpu_info:
            print(cpu_info['model name'])
        output = subprocess.check_output("cat /proc/meminfo", shell=True).strip().decode().split('\n')
        mem_info = {item[0]: item[1] for item in [re.split("\s*:\s*", line, maxsplit=2) for line in output if line]}
        if 'MemTotal' in mem_info:
            ram = mem_info['MemTotal'].split()
            count = 0
            while int(ram[0]) >= 1024:
                count += 1
                ram[0] = int(ram[0]) / 1024
            print("Memory: " + str("%.2f" % round(ram[0],2)) + " " + ("kB" if count == 0 else "MB" if count == 1 else "GB"))
    print("Start load: " + f"{psutil.getloadavg()[0] / psutil.cpu_count() * 100:.2f}%")
    print("---------------------------")


def log_to_file(dataset, score, emission, model):
    """Output models and scores to a csv file."""
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, 'w') as file:
            file.write(','.join(COLUMN_NAMES))

    with open(FILE_NAME, 'a') as file:
        file.write(f"\n{dataset},{model},")
        file.write(str(psutil.getloadavg()[0] / psutil.cpu_count() * 100))
        file.write(f",{score},{emission.duration},{emission.emissions},{emission.energy_consumed}")
