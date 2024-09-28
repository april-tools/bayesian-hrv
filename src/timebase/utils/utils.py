import copy
import csv
import itertools
import math
import random
import subprocess
import typing as t

from timebase.data.static import *
from timebase.utils import yaml


def set_random_seed(seed: int, verbose: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    if verbose > 2:
        print(f"set random seed: {seed}")


def update_dict(target: t.Dict, source: t.Dict, replace: bool = False):
    """add or update items in source to target"""
    for key, value in source.items():
        if replace:
            target[key] = value
        else:
            if key not in target:
                target[key] = []
            target[key].append(value)


def check_output(command: list):
    """Execute command in subprocess and return output as string"""
    return subprocess.check_output(command).strip().decode()


def save_args(args):
    """Save args object as dictionary to output_dir/args.yaml"""
    """Save args object as dictionary to args.output_dir/args.yaml"""
    arguments = copy.deepcopy(args.__dict__)
    try:
        arguments["git_hash"] = check_output(["git", "describe", "--always"])
        arguments["hostname"] = check_output(["hostname"])
    except subprocess.CalledProcessError as e:
        if args.verbose:
            print(f"Unable to call subprocess: {e}")
    yaml.save(filename=os.path.join(args.output_dir, "args.yaml"), data=arguments)


def write_csv(output_dir, content: list):
    with open(os.path.join(output_dir, "results.csv"), "a") as file:
        writer = csv.writer(file)
        writer.writerow(content)


def get_sequences_boundaries_index(arr, value):
    """
    Given a 1D array-like object return a list of lists where each sub-list
    contains two elements, the former being the index where a given sequence of
    array entries equal to value starts and the letter being the index where
    the same sequence ends
    For example:
    arr = np.array([0,0,1,1,1,0,0,1,0,1])
    get_indexes(arr=arr, value=1) -> [[2, 4], [7, 7], [9, 9]]
    get_indexes(arr=arr, value=0) -> [[0, 1], [5, 6], [8, 8]]
    """
    seqs = [(key, len(list(val))) for key, val in itertools.groupby(arr)]
    seqs = [
        (key, sum(s[1] for s in seqs[:i]), len) for i, (key, len) in enumerate(seqs)
    ]
    return [[s[1], s[1] + s[2] - 1] for s in seqs if s[0] == value]


def generatePrime(n):
    primes = []
    X = 0
    i = 2
    flag = False
    while X < n:
        flag = True
        for j in range(2, math.floor(math.sqrt(i)) + 1):
            if i % j == 0:
                flag = False
                break
        if flag:
            primes.append(i)
            X += 1
        i += 1
    return primes
