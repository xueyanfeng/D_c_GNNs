import numpy as np
import matplotlib.pyplot as plt

group_number = 8
datasets = ["Cornell", "Texas", "Wisconsin", "Chameleon"]
GNNs = ["GCN", "GAT", "MixHop", "GCNII", "PPNP", "DropEdge", "Grand", "AUTOGNN"]

A = {
    "Cornell": [49.19, 52.97, 66.49, 60.27, 55.95, 52.16, 58.11, 54.59],
    "Texas": [48.11, 58.65, 72.97, 57.57, 54.86, 65.41, 54.05, 54.59],
    "Wisconsin": [50.20, 53.14, 72.16, 57.25, 43.92, 55.49, 49.41, 47.65],
    "Chameleon": [61.34, 59.43, 60.35, 53.44, 52.21, 59.82, 43.33, 59.56],
}
A_err = {
    "Cornell": [4.15, 4.86, 3.46, 3.21, 5.55, 2.72, 1.81, 1.62],
    "Texas": [3.78, 6.05, 4.83, 2.43, 4.84, 6.02, 1.21, 1.62],
    "Wisconsin": [4.90, 2.05, 4.09, 4.37, 7.08, 2.49, 3.59, 3.93],
    "Chameleon": [0.57, 1.20, 0.76, 0.97, 1.12, 0.52, 2.89, 1.01],
}


A1 = {
    "Cornell": [94.05, 91.35, 94.05, 92.97, 91.35, 91.35, 88.65, 94.32],
    "Texas": [84.05, 86.22, 86.49,  88.11, 86.76, 83.24, 88.65, 85.68],
    "Wisconsin": [95.10, 95.10, 94.12, 96.08, 94.31, 93.73, 94.12, 93.73],
    "Chameleon": [66.43, 69.01, 67.50, 65.92, 64.96, 62.17, 63.05, 68.90],
}
A1_err = {
    "Cornell": [1.62, 2.36, 1.08, 1.79, 2.65, 1.08, 1.08, 0.81],
    "Texas": [3.07, 2.82, 1.21, 1.32, 2.25, 4.80, 1.62, 2.11],
    "Wisconsin": [1.32, 0.98, 1.24, 0.00, 1.63, 1.18, 0.00, 2.11],
    "Chameleon": [0.62, 1.08, 1.08, 1.13, 0.45, 0.64, 0.96, 0.94],
}

MLP = [69.46, 69.46, 77.65, 46.75]
MLP_err = [3.43, 3.43, 2.51, 1.34]

total_width = 2
width = total_width / group_number
err_attr = {"elinewidth": 2, "ecolor": "black", "capsize": 3}  # This is the property of the error bars

fig, ax = plt.subplots(2, 2, figsize=(25, 15))

position = [[0, 0], [0, 1], [1, 0], [1, 1]]
for i in range(4):
    row, col = position[i]
    x = np.linspace(
        start=1, stop=group_number, endpoint=True, num=group_number, dtype=np.int32
    )
    ax[row, col].bar(
        x - 0.5 * width,
        A[datasets[i]],
        yerr=A_err[datasets[i]],
        error_kw=err_attr,
        width=width,
        color="b",
        label='$\mathcal{E}$',
    )
    ax[row, col].bar(
        x + 0.5 * width,
        A1[datasets[i]],
        yerr=A1_err[datasets[i]],
        error_kw=err_attr,
        width=width,
        color="m",
        label=r"$\widetilde{\mathcal{E}}$",
    )
    ax[row, col].hlines(
        y=MLP[i], xmin=0, xmax=8.5, linewidth=MLP_err[i], colors="gray", label="MLP"
    )
    ax[row, col].set_xlabel(datasets[i], color="black", fontsize=15)
    ax[row, col].set_ylabel(r"Test Accuracy(%)", size=15, color="black")
    ax[row, col].tick_params(labelsize=9)
    ax[row, col].set_xlim(0.5, 8.5)
    ax[row, col].set_ylim(0, 100)
    ax[row, col].spines["right"].set_color("none")
    ax[row, col].spines["top"].set_color("none")
    ax[row, col].spines["bottom"].set_color("none")
    ax[row, col].spines["left"].set_color("none")
    ax[row, col].set_xticks(x)
    ax[row, col].set_xticklabels(labels=GNNs, size=12, rotation=0)  #, ha='right'
    ax[row, col].set_yticks(
        ticks=np.linspace(start=0, stop=100, endpoint=True, num=11, dtype=np.float32)
    )
    
lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(
    lines,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.95),
    fontsize=15,
)

plt.savefig(fname="bar.pdf")
plt.show()