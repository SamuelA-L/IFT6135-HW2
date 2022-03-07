import matplotlib.pyplot as plt
import numpy as np

def read_file_values(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()

    return [float(line) for line in lines]


def plot(title, arrays, y_label, save=False):

    for i in range(len(arrays)):
        plt.plot(arrays[i], label='experiment ' + str(i+1))
    plt.legend()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.xticks(np.arange(0, len(arrays[i]), 1))

    if save:
        plt.savefig('/home/samuel/Desktop/figures_HW2/' + title + '.png')

    plt.show()




train_losses_lstm = []
train_ppl_lstm = []
valid_losses_lstm = []
valid_ppl_lstm = []

train_losses_vit = []
train_ppl_vit = []
valid_losses_vit = []
valid_ppl_vit = []


for i in range(6):
    train_losses_lstm.append(read_file_values('logs/' + str(i+1) + '/train_loss.txt'))
    train_ppl_lstm.append(read_file_values('logs/' + str(i+1) + '/train_ppl.txt'))
    valid_losses_lstm.append(read_file_values('logs/' + str(i+1) + '/valid_loss.txt'))
    valid_ppl_lstm.append(read_file_values('logs/' + str(i+1) + '/valid_ppl.txt'))


plot("LSTM train loss by experiment", train_losses_lstm, 'loss', save=False)
plot("LSTM validation loss by experiment", valid_losses_lstm, 'loss', save=False)
plot("LSTM train ppl by experiment", train_ppl_lstm, 'loss', save=False)
plot("LSTM valid ppl by experiment", valid_losses_lstm, 'loss', save=False)
