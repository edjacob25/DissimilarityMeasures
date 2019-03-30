import argparse
import configparser
import json
import math
import multiprocessing
import os
import requests
import subprocess
import time
from openpyxl import Workbook
from shutil import copyfile


def get_number_of_clusters(filepath: str):
    with open(filepath) as file:
        for line in file:
            line_upper = line.upper()
            if line_upper.startswith("@ATTRIBUTE") and ("CLASS" in line_upper or "CLUSTER" in line_upper):
                clusters = line.split(" ", 2)[-1]
                return len(clusters.split(","))
            if line_upper.startswith("@DATA"):
                raise Exception("Could not found Class or Cluster attribute")


def remove_attribute(filepath: str, attribute: str):
    attributes = []
    data = []
    permitted_indexes = []
    with open(filepath) as file:
        relation_name = file.readline()
        data_processing = False
        i = 0
        for line in file:
            if not data_processing:
                line_upper = line.upper()
                if line_upper.startswith("@ATTRIBUTE"):
                    if attribute.upper() not in line_upper:
                        attributes.append(line)
                        permitted_indexes.append(i)
                    i += 1
                    continue
                if line_upper.startswith("@DATA"):
                    data_processing = True
            else:
                line_data = [x.lstrip() for x in line.split(',')]
                data.append(line_data)

    with open(filepath, "w") as new_file:
        new_file.write(relation_name)
        for item in attributes:
            new_file.write(item)
        new_file.write("\n")
        new_file.write("@DATA\n")
        for datapoint in data:
            points = [x for i, x in enumerate(datapoint) if i in permitted_indexes]
            result = ",".join(points)
            new_file.write(result)


def cluster_dataset(filepath: str, classpath: str = None, no_classpath: bool = False, verbose: bool = False):
    clustered_file_path = filepath.replace(".arff", "_clustered.arff")
    command = ["java", "-Xmx8192m"]
    if not no_classpath:
        java_classpath = "/mnt/c/Program Files/Weka-3-9/weka.jar:/home/jacob/wekafiles/packages" \
                         "/SimmilarityMeasuresForCategoricalData/DissimilarityMeasures-0.1.jar"
        if classpath is not None:
            java_classpath = classpath
        command.append("-cp")
        command.append(java_classpath)

    command.append("weka.filters.unsupervised.attribute.AddCluster")
    command.append("-W")

    num_clusters = get_number_of_clusters(filepath)
    num_procs = multiprocessing.cpu_count()
    distance_function = "\"weka.core.LearningBasedDissimilarity -R first-last\""
    clusterer = "weka.clusterers.CategoricalKMeans -init 1 -max-candidates 100 -periodic-pruning 10000 " \
        f"-min-density 2.0 -t1 -1.25 -t2 -1.0 -N {num_clusters} -A {distance_function} -I 500 " \
        f"-num-slots {math.floor(num_procs / 3)} -S 10"
    command.append(clusterer)
    command.append("-i")
    command.append(filepath)
    command.append("-o")
    command.append(clustered_file_path)
    command.append("-I")
    command.append("Last")

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(result.stderr.decode("utf-8"))
    if verbose:
        print(result.stdout.decode("utf-8"))

    if "Exception" not in result.stderr.decode("utf-8"):
        remove_attribute(clustered_file_path, "Class")
        print(f"Finished clustering dataset {filepath}")
    else:
        if os.path.exists(clustered_file_path):
            os.remove(clustered_file_path)
        raise Exception(f"There was a error running weka with the file {filepath.rsplit('/')[-1]} and the " +
                        f"following command {' '.join(result.args)}")


def copy_files(filepath: str):
    path, file = filepath.rsplit("/", 1)
    filename = file.split(".")[0]
    os.mkdir(f"{path}/{filename}")
    new_filepath = f"{path}/{filename}/{file}"
    copyfile(filepath, new_filepath)

    new_clustered_filepath = f"{path}/{filename}/{filename}.clus"
    copyfile(f"{path}/{filename}_clustered.arff", new_clustered_filepath)

    return new_filepath, new_clustered_filepath


def get_f_measure(filepath: str, clustered_filepath: str, exe_path: str = None) -> str:
    command = ["MeasuresComparator.exe", "-c", clustered_filepath, "-r", filepath]
    if exe_path is not None:
        command[0] = exe_path
    result = subprocess.run(command, stdout=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Could not get F-Measure\nError -> {result.stdout.decode('utf-8')}")
        raise Exception("Could not calculate f-measure")
    else:
        return result.stdout.decode('utf-8')


def send_notification(message: str, title: str):
    config = configparser.ConfigParser()
    config.read("config.ini")
    data = {"body":message, "title": title, "type": "note"}
    headers = {"Content-Type": "application/json", "Access-Token": config["SECRETS"]["Pushbullet_token"]}
    r = requests.post("https://api.pushbullet.com/v2/pushes", headers=headers, data=json.dumps(data))


parser = argparse.ArgumentParser(description='Does the analysis of a directory containing categorical datasets')
parser.add_argument('directory', help="Directory in which the cleaned datasets are")
parser.add_argument('-cp', help="Classpath for the weka invocation, needs to contain the weka.jar file and probably "
                                "the jar of the measure ")
parser.add_argument("-v", "--verbose", help="Show the output of the weka commands", action='store_true')
parser.add_argument("-f", "--measure-calc", help="Path to the f-measure calculator", dest='measure_calculator_path')

args = parser.parse_args()

if not os.path.isdir(args.directory):
    print("The selected path is not a directory")
    exit(1)

workbook = Workbook()
ws = workbook.active
ws['B1'] = "Option 1"
ws.merge_cells('B1:C1')
ws['D1'] = "Option 2"
ws.merge_cells('D1:E1')
ws['F1'] = "Option 3"
ws.merge_cells('F1:G1')
ws['H1'] = "Option 4"
ws.merge_cells('H1:I1')
ws['J1'] = "Option 5"
ws.merge_cells('J1:K1')
ws['L1'] = "Option 6"
ws.merge_cells('L1:M1')

start = time.time()

root_dir = os.path.abspath(args.directory)
index = 2
for item in os.listdir(root_dir):
    if item.rsplit('.', 1)[-1] == "arff" and "clustered" not in item:
        item_fullpath = os.path.join(root_dir, item)
        try:
            cluster_dataset(item_fullpath, verbose=args.verbose, classpath=args.cp)
            new_filepath, new_clustered_filepath = copy_files(item_fullpath)
            f_measure = get_f_measure(new_filepath, new_clustered_filepath, exe_path=args.measure_calculator_path)
            ws.cell(row=index, column=1, value=item)
            ws.cell(row=index, column=2, value=f_measure.rstrip())
            index += 1
        except KeyboardInterrupt:
            print(f"The analysis of the file {item} was requested to be finished by using Ctrl-C")
            continue
        except Exception as exc:
            print(exc)
            print(f"Skipping file {item}")
            continue
        finally:
            print("\n\n")

end = time.end()
# cluster_dataset("/mnt/f/Datasets/cars.arff", verbose=args.verbose)
# new_filepath, new_clustered_filepath = copy_files("/mnt/f/Datasets/cars.arff")
# f_measure = get_f_measure(new_filepath, new_clustered_filepath,
#                           exe_path="/home/jacob/Projects/MeasuresComparator/MeasuresComparator/bin/Release"
#                                    "/netcoreapp2.1/linux-x64/publish/MeasuresComparator")
# print(f_measure)
# ws.cell(row=2, column=1, value="cars")
# ws.cell(row=2, column=2, value=f_measure.rstrip())
# print("Going to save")
workbook.save(filename=f"{args.directory}/Results.xlsx")
send_notification(f"It took {end - start} and processed {index - 2} datasets","Analysis finished")
