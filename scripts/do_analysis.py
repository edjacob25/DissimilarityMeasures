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


# TODO: Add option to run other dissimilarity measures
# TODO: Add option to read classpath from the config file
def cluster_dataset(filepath: str, classpath: str = None, no_classpath: bool = False, verbose: bool = False,
                    strategy: str = "A", weight_strategy: str = "N", other_measure: str = None, start_mode: str = "1"):
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
    if verbose:
        print(f"Number of clusters for {filepath} is {num_clusters}")
    num_procs = multiprocessing.cpu_count()
    distance_function = f"\"weka.core.LearningBasedDissimilarity -R first-last -S {strategy} -w {weight_strategy}\""
    if other_measure is not None:
        distance_function = f"\"{other_measure}\""
    clusterer = f"weka.clusterers.CategoricalKMeans -init {start_mode} -max-candidates 100 -periodic-pruning 10000 " \
        f"-min-density 2.0 -t1 -1.25 -t2 -1.0 -N {num_clusters} -A {distance_function} -I 500 " \
        f"-num-slots {math.floor(num_procs / 3)} -S 10"
    command.append(clusterer)
    command.append("-i")
    command.append(filepath)
    command.append("-o")
    command.append(clustered_file_path)
    command.append("-I")
    command.append("Last")

    start = time.time()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end = time.time()

    print(result.stderr.decode("utf-8"))
    if verbose:
        print(result.stdout.decode("utf-8"))
        print(f"Analyzing dataset {filepath} with strategy {strategy} took {end - start}")

    if "Exception" not in result.stderr.decode("utf-8"):
        remove_attribute(clustered_file_path, "Class")
        print(f"Finished clustering dataset {filepath} with strategy {strategy} and weight {weight_strategy}")
    else:
        if os.path.exists(clustered_file_path):
            os.remove(clustered_file_path)

        if start_mode == "1":
            print("There was an error running weka with the k-means++ mode, trying with classic mode")
            cluster_dataset(filepath, classpath=classpath, no_classpath=no_classpath, verbose=verbose,
                            strategy=strategy, weight_strategy=weight_strategy, other_measure=other_measure,
                            start_mode="0")
        else:
            raise Exception(f"There was a error running weka with the file {filepath.rsplit('/')[-1]} and the " +
                        f"following command {' '.join(result.args)}")


def copy_files(filepath: str, strategy: str = "", weight_strategy: str = ""):
    path, file = filepath.rsplit("/", 1)
    filename = file.split(".")[0]
    os.mkdir(f"{path}/{filename}_{strategy}_{weight_strategy}")
    new_filepath = f"{path}/{filename}_{strategy}_{weight_strategy}/{file}"
    copyfile(filepath, new_filepath)

    new_clustered_filepath = f"{path}/{filename}_{strategy}_{weight_strategy}/{filename}.clus"
    copyfile(f"{path}/{filename}_clustered.arff", new_clustered_filepath)

    return new_filepath, new_clustered_filepath


def get_f_measure(filepath: str, clustered_filepath: str, exe_path: str = None, verbose: bool = False) -> str:
    command = ["MeasuresComparator.exe", "-c", clustered_filepath, "-r", filepath]
    if exe_path is not None:
        command[0] = exe_path
    start = time.time()
    result = subprocess.run(command, stdout=subprocess.PIPE)
    end = time.time()
    text_result = result.stdout.decode('utf-8')

    if result.returncode != 0:
        print(f"Could not get F-Measure\nError -> {text_result}")
        raise Exception("Could not calculate f-measure")
    else:
        if verbose:
            print(f"Calculating f-measure took {end - start}")
        print(f"Finished getting f-measure for {filepath}, f-measure -> {text_result}")
        return text_result


def send_notification(message: str, title: str):
    config = configparser.ConfigParser()
    config.read("config.ini")
    data = {"body": message, "title": title, "type": "note"}
    headers = {"Content-Type": "application/json", "Access-Token": config["SECRETS"]["Pushbullet_token"]}
    requests.post("https://api.pushbullet.com/v2/pushes", headers=headers, data=json.dumps(data))


def format_seconds(seconds: float) -> str:
    seconds = math.fabs(seconds)
    if seconds > 3600:
        return f"{seconds / 3600} hours"
    elif seconds > 60:
        return f"{seconds / 60} minutes"
    else:
        return f"{seconds} seconds"


def do_analysis(directory: str, verbose: bool, cp: str = None, measure_calculator_path: str = None):
    workbook = Workbook()
    ws = workbook.active

    i = 2
    for strategy in ["A", "B", "C", "D", "E", "None"]:
        for weight in ["Normal", "Kappa", "Aug"]:
            ws.cell(column=i, row=1, value=f"Strategy {strategy} with weight {weight}")
            i += 1

    start = time.time()
    root_dir = os.path.abspath(directory)
    index = 2
    for item in os.listdir(root_dir):
        if item.rsplit('.', 1)[-1] == "arff" and "clustered" not in item:
            item_fullpath = os.path.join(root_dir, item)
            try:
                column = 2
                ws.cell(row=index, column=1, value=item)
                for strategy in ["A", "B", "C", "D", "E", "N"]:
                    for weight in ["N", "K", "A"]:
                        cluster_dataset(item_fullpath, verbose=verbose, classpath=cp, strategy=strategy,
                                        weight_strategy=weight)
                        new_filepath, new_clustered_filepath = copy_files(item_fullpath, strategy=strategy,
                                                                          weight_strategy=weight)
                        f_measure = get_f_measure(new_filepath, new_clustered_filepath,
                                                  exe_path=measure_calculator_path,
                                                  verbose=verbose)
                        ws.cell(row=index, column=column, value=float(f_measure))
                        column += 1

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

    end = time.time()
    workbook.save(filename=f"{directory}/Results.xlsx")

    time_str = format_seconds(end - start)
    send_notification(f"It took {time_str} and processed {index - 2} datasets", "Analysis finished")


def do_analysis_2(directory: str, verbose: bool, cp: str = None, measure_calculator_path: str = None):
    workbook = Workbook()
    ws = workbook.active

    i = 2
    for measure in ["Eskin", "Gambaryan", "Goodall", "Lin", "OccurenceFrequency", "InverseOccurenceFrequency",
                    "Euclidian", "Manhattan", "LinModified", "LinModified2"]:
        ws.cell(column=i, row=1, value=f"Measure {measure}")
        i += 1

    start = time.time()
    root_dir = os.path.abspath(directory)
    index = 2
    for item in os.listdir(root_dir):
        if item.rsplit('.', 1)[-1] == "arff" and "clustered" not in item:
            item_fullpath = os.path.join(root_dir, item)
            try:
                column = 2
                ws.cell(row=index, column=1, value=item)
                measures = ["weka.core.Eskin", "weka.core.Gambaryan", "weka.core.Goodall", "weka.core.Lin",
                            "weka.core.OccurenceFrequency", "weka.core.InverseOccurenceFrequency",
                            "weka.core.EuclideanDistance", "weka.core.ManhattanDistance", "weka.core.LinModified",
                            "weka.core.LinModified2"]
                for measure in measures:
                    cluster_dataset(item_fullpath, verbose=verbose, classpath=cp, other_measure=measure)
                    new_filepath, new_clustered_filepath = copy_files(item_fullpath, strategy=measure)
                    f_measure = get_f_measure(new_filepath, new_clustered_filepath, exe_path=measure_calculator_path,
                                              verbose=verbose)
                    ws.cell(row=index, column=column, value=float(f_measure))
                    column += 1

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

    end = time.time()
    workbook.save(filename=f"{directory}/Results_Other.xlsx")

    time_str = format_seconds(end - start)
    send_notification(f"It took {time_str} and processed {index - 2} datasets", "Analysis finished")


parser = argparse.ArgumentParser(description='Does the analysis of a directory containing categorical datasets')
parser.add_argument('directory', help="Directory in which the cleaned datasets are")
parser.add_argument('-cp', help="Classpath for the weka invocation, needs to contain the weka.jar file and probably "
                                "the jar of the measure ")
parser.add_argument("-v", "--verbose", help="Show the output of the weka commands", action='store_true')
parser.add_argument("-f", "--measure-calc", help="Path to the f-measure calculator", dest='measure_calculator_path')
parser.add_argument("--alternate-analysis", help="Does the alternate analysis with the already known simmilarity "
                                                 "measures", action='store_true')
parser.add_argument("-s", "--save", help="Path to the f-measure calculator", action='store_true')
# TODO: Actually save the output of the commands


args = parser.parse_args()

# TODO: Read a single file
if not os.path.isdir(args.directory):
    print("The selected path is not a directory")
    exit(1)

if not args.alternate_analysis:
    do_analysis(args.directory, args.verbose, args.cp, args.measure_calculator_path)
else:
    do_analysis_2(args.directory, args.verbose, args.cp, args.measure_calculator_path)
