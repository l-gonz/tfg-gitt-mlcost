import platform
import subprocess
import re
import os
import psutil


FILE_NAME = 'output.csv'

INFO_COLUMN_NAMES = ["dataset", "cpu_load", ""]
MODEL_COLUMN_NAMES = ["accuracy", "time", "emissions"]


def print_output(name, score, emissions, time):
    """Print model scores to standard output."""
    print("---------------------------")
    print(name)
    print(f'Accuracy: {score:.4f}')
    print(f"Emissions: {emissions:.4e}kg (CO2 equ)")
    print(f"Time: {time:.4f}s")


def print_computer_info():
    """Print platform, CPU and RAM info to standard output."""
    print("Running on " + platform.node())
    print(platform.freedesktop_os_release()['PRETTY_NAME'] + " " + platform.machine())
    print("Python " + platform.python_version())
    if platform.system() == "Linux":
        output = subprocess.check_output("cat /proc/cpuinfo", shell=True).strip().decode().split('\n')
        cpu_info = {item[0]: item[1] for item in [re.split("\s*:\s*", line, maxsplit=2) for line in output]}
        if 'model name' in cpu_info:
            print(cpu_info['model name'])
        output = subprocess.check_output("cat /proc/meminfo", shell=True).strip().decode().split('\n')
        mem_info = {item[0]: item[1] for item in [re.split("\s*:\s*", line, maxsplit=2) for line in output]}
        if 'MemTotal' in mem_info:
            ram = mem_info['MemTotal'].split()
            count = 0
            while int(ram[0]) >= 1024:
                count += 1
                ram[0] = int(ram[0]) / 1024
            print("Memory: " + str("%.2f" % round(ram[0],2)) + " " + ("kB" if count == 0 else "MB" if count == 1 else "GB"))


def get_column_names(model_name):
    """Return column names for the given model as comma-separated values."""
    return ",".join([model_name + "_" + column for column in MODEL_COLUMN_NAMES])


def log_to_file(dataset, scores, emissions, time, models):
    """Output models and scores to a csv file."""
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, 'w') as file:
            file.write(','.join(INFO_COLUMN_NAMES))
            file.write(','.join([get_column_names(name) for name in models.keys()]))

    with open(FILE_NAME, 'a') as file:
        file.write('\n')
        file.write(dataset if dataset else "iris" + ',')
        file.write(str(psutil.getloadavg()[0] / psutil.cpu_count() * 100) + ',')
        file.write(','.join([f"{scores[name]},{time[name]},{emissions[name]}" for name in models.keys()]))