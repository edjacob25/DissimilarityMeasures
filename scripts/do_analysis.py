import argparse
import math
import multiprocessing
import os
import subprocess
from shutil import copyfile


def get_number_of_clusters(file_path: str):
    with open(file_path) as file:
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


def analyze_dataset(filepath: str, classpath: str = None, no_classpath: bool = False, verbose: bool = False):
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
        print(f"Finished analyzing dataset {filepath}")
    else:
        if os.path.exists(clustered_file_path):
            os.remove(clustered_file_path)
        raise Exception(f"There was a error running weka with the file {filepath.rsplit('/')[-1]} and the " +
                        f"following command {' '.join(result.args)}")


def copy_files(filepath: str):
    path, file = filepath.rsplit("/", 1)
    filename = file.split(".")[0]
    os.mkdir(f"{path}/{filename}")
    copyfile(filepath, f"{path}/{filename}/{file}")
    copyfile(f"{path}/{filename}_clustered.arff", f"{path}/{filename}/{filename}.clus")


def get_f_measure(filepath: str):
    pass


parser = argparse.ArgumentParser(description='Does the analysis of a directory containing categorical datasets')
parser.add_argument('directory', help="Directory in which the cleaned datasets are")
parser.add_argument('-cp', help="Classpath for the weka invocation, needs to contain the weka.jar file and probably "
                                "the jar of the measure ")
parser.add_argument("-v", "--verbose", help="Show the output of the weka commands", action='store_true')

args = parser.parse_args()

if not os.path.isdir(args.directory):
    print("The selected path is not a directory")
    exit(1)

root_dir = os.path.abspath(args.directory)
for item in os.listdir(root_dir):
    if item.rsplit('.', 1)[-1] == "arff" and "clustered" not in item:

        item = os.path.join(root_dir, item)
        try:
            pass
            analyze_dataset(item, verbose=args.verbose, classpath=args.cp)
            # copy_files(item)

        except Exception as exc:
            print(exc)
            print(f"Skipping file {item}")
            continue
        finally:
            print("\n\n")

# analyze_dataset("/mnt/f/Datasets/car.arff", verbose=True)
