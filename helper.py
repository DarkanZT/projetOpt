# Mostly from https://github.com/patrickloeber/snake-ai-pytorch/blob/main/helper.py

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

plt.ion()

def plot(errors):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Absolute errors progression')
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.ylim(ymin=0)
    plt.text(len(errors)-1, errors[-1], str(errors[-1]))
    plt.show(block=False)
    plt.pause(.1)

def save_plot(save_path):
    plt.savefig(save_path, format="png")  # Save as PNG
    print(f"Plot saved as '{save_path}'.")