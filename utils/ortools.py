"""
OR-Tools integration utilities for dataset preparation and data generation in CMIP.
"""
from utils.vrp import entrance
import torch
import time
import os
import sys

recordfile = os.path.join(os.getcwd(), "timeRecord.txt")


def prepare_dir(objective: str, anum: int, cnum: int, timilimitation: int = None) -> str:
    """
    Prepare and return a directory path for saving OR-Tools data, creating directories as needed.
    Args:
        objective: Objective name
        anum: Number of agents
        cnum: Number of cities
        timilimitation: Optional time limit
    Returns:
        Path to the prepared directory
    """
    datapath = os.path.join(os.getcwd(), "../dataset")
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    datapath = os.path.join(datapath, "ortools")
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    if timilimitation is not None:
        datapath = os.path.join(datapath, "TL{}".format(timilimitation))
        try:
            os.stat(datapath)
        except:
            os.mkdir(datapath)

    datapath = os.path.join(datapath, objective)
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    datapath = os.path.join(datapath, "agent{}".format(anum))
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    datapath = os.path.join(datapath, "city{}".format(cnum))
    try:
        os.stat(datapath)
    except:
        os.mkdir(datapath)

    return datapath


def generate_data(argv) -> None:
    """
    Generate and save OR-Tools data for a given set of parameters.
    Args:
        argv: Command-line arguments
    """
    if len(argv) < 4:
        print("objective, anum, cnum, dnum, timelimitation(s)")
    # objective = argv[1]
    # anum = int(argv[2])  # agent number
    # cnum = int(argv[3])  # city number
    # dnum = int(argv[4])  # data order
    # timeLimitation = int(argv[5])
    objective = "test"
    anum = 5
    cnum = 100
    dnum = 1000
    timeLimitation = 10
    timeLimitation = None
    datapath = prepare_dir(objective, anum, cnum, timeLimitation)
    filename = os.path.join(
        datapath,
        "minmax_ortools_{}_agent{}_city{}_num{}.pt".format(objective, anum, cnum, dnum),
    )
    start_time = time.time()
    if timeLimitation is not None:
        tourlen, coords = entrance(cnum, anum, timeLimitation)
    else:
        tourlen, coords = entrance(cnum, anum)
    duration = time.time() - start_time
    dataset = {"cities": coords, "tourlen": tourlen, "time": duration}
    torch.save(dataset, filename)
    print("save ", filename)
    print(tourlen, duration)


if __name__ == "__main__":
    generate_data(sys.argv)
