import matplotlib.pyplot as plt
import re

with open("trainingLogs_2024-04-21_22-28-06.log", "r") as f:
    lines = f.readlines()

    mapped = map(lambda line: 
        re.findall(
        r"((?:[0-9]+)(?:.[0-9]+)?)", line),
        lines)

    x = []
    tl = []
    vl = [] 

    for line in mapped:
        try:
            x.append(float(line[0]))
            tl.append(float(line[1]))
            vl.append(float(line[2]))
        except IndexError:
            pass

    plt.plot(x, tl, label="train loss")
    plt.plot(x, vl, label="val loss")

    plt.legend()
    plt.show()