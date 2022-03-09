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

def plot_dual(title, arrays, series, y_label, save=False):

    for i in range(len(arrays)):
        plt.plot(np.cumsum(np.array(series[i])), arrays[i], label='experiment ' + str(i+1))
    plt.legend()
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(y_label)

    if save:
        plt.savefig('/home/samuel/Desktop/figures_HW2/' + title + '.png')

    plt.show()


train_losses_lstm = []
train_ppl_lstm = []
train_time_lstm = []
valid_losses_lstm = []
valid_ppl_lstm = []


for i in range(1, 7):
    train_losses_lstm.append(read_file_values('logs/' + str(i) + '/train_loss.txt'))
    train_ppl_lstm.append(read_file_values('logs/' + str(i) + '/train_ppl.txt'))
    valid_losses_lstm.append(read_file_values('logs/' + str(i) + '/valid_loss.txt'))
    valid_ppl_lstm.append(read_file_values('logs/' + str(i) + '/valid_ppl.txt'))


plot("LSTM train loss by experiment", train_losses_lstm, 'loss', save=False)
plot("LSTM validation loss by experiment", valid_losses_lstm, 'loss', save=False)
plot("LSTM train ppl by experiment", train_ppl_lstm, 'loss', save=False)
plot("LSTM valid ppl by experiment", valid_losses_lstm, 'loss', save=False)


train_losses_vit = []
train_accs_vit = []
train_time_vit = []
valid_losses_vit = []
valid_accs_vit = []
valid_time_vit = []

for i in range(7, 14):
    train_losses_vit.append(read_file_values('logs/' + str(i) + '/train_loss.txt'))
    train_accs_vit.append(read_file_values('logs/' + str(i) + '/train_accs.txt'))
    train_time_vit.append(read_file_values('logs/' + str(i) + '/train_time.txt'))
    valid_losses_vit.append(read_file_values('logs/' + str(i) + '/valid_loss.txt'))
    valid_accs_vit.append(read_file_values('logs/' + str(i) + '/valid_accs.txt'))
    valid_time_vit.append(read_file_values('logs/' + str(i) + '/valid_time.txt'))


plot("Vision Transformer train accuracy over epoch by experiment", train_accs_vit, 'Accuracy', save=True)
plot_dual("Vision Transformer train accuracy over wall-clock time by experiment", train_accs_vit, train_time_vit, 'Accuracy', save=True)
plot("Vision Transformer valid accuracy over epoch by experiment", valid_accs_vit, 'Accuracy', save=True)
plot_dual("Vision Transformer valid accuracy over wall-clock by experiment", valid_accs_vit, valid_time_vit, 'Accuracy', save=True)
