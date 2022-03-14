import glob
import csv
import os.path

import matplotlib.pyplot as plt

if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111)
    list_path_csv = glob.glob("result/*.csv")
    print("num motion: {}".format(len(list_path_csv)))
    list_xy = []
    for path_csv in list_path_csv:
        if path_csv.endswith("_fix.csv") or path_csv.endswith("_svd.csv"):
            continue
        file_name = os.path.basename(path_csv).split(".")[0]
        print(path_csv, file_name)
        with open(path_csv) as f:
            reader = csv.reader(f)
            list_x0 = []
            list_y0 = []
            for row in reader:
                ratio = float(row[1])
                err = float(row[2])
                list_x0.append(ratio)
                list_y0.append(err)
            if len(list_x0) < 11:
                continue
            list_xy.append(
                [file_name,list_x0[6], list_y0[6], list_x0[10], list_y0[10]]
            )

    with open('convergence.csv', 'w') as f:
        for xy in list_xy:
            f.write("{},{},{},{},{}\n".format(
                xy[0], xy[1], xy[2], xy[3], xy[4]))

    '''
    ax.scatter(list_xy6[0], list_xy6[1], s=10.0, marker="D")
    #ax.scatter(list_xy8[0], list_xy8[1], s=10.0, marker="s")
    ax.scatter(list_xy10[0], list_xy10[1], s=10.0, marker="o")
    plt.yscale('log')
    plt.show()
    '''



