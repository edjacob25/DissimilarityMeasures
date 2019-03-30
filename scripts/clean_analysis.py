import argparse
import os
from shutil import rmtree

parser = argparse.ArgumentParser(description='Does the analysis of a directory containing categorical datasets')
parser.add_argument('directory', help="Directory in which the cleaned datasets are")

args = parser.parse_args()

if not os.path.isdir(args.directory):
    print("The selected path is not a directory")
    exit(1)

root_dir = os.path.abspath(args.directory)
for item in os.listdir(root_dir):
    item_full_path = os.path.join(root_dir, item)
    if os.path.isdir(item_full_path):
        rmtree(item_full_path)
        continue
    if "clustered" in item:
        os.remove(item_full_path)