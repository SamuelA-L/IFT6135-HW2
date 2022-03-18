import matplotlib.pyplot as plt
import numpy as np

def read_file_values(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()

    return [float(line) for line in lines]


def plot(title, arrays, y_label, save=False, plot=True):

    for i in range(len(arrays)):
        plt.plot(arrays[i], label='experiment ' + str(i+1))
    plt.legend()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.xticks(np.arange(0, len(arrays[i]), 1))

    if save:
        plt.savefig('/home/samuel/Desktop/figures_HW2/' + title + '.png')
    if plot:
        plt.show()


def plot_dual(title, arrays, series, y_label, save=False, plot=True):

    for i in range(len(arrays)):
        plt.plot(np.cumsum(np.array(series[i])), arrays[i], label='experiment ' + str(i+1))
    plt.legend()
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(y_label)

    if save:
        plt.savefig('/home/samuel/Desktop/figures_HW2/' + title + '.png')
    if plot:
        plt.show()

    plt.close

def bar_plot(title, array, x_label, y_label, save=False, plot=True):

    plt.bar(np.arange(1, len(array)+1), array)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if save:
        plt.savefig('/home/samuel/Desktop/figures_HW2/' + title + '.png')
    if plot:
        plt.show()

    plt.close()


SAVE_FIGURES_LSTM = False
PLOT_FIGURES_LSTM = False

SAVE_FIGURES_ViT = True
PLOT_FIGURES_ViT = True

train_losses_lstm = []
train_ppl_lstm = []
train_time_lstm = []
valid_losses_lstm = []
valid_ppl_lstm = []

test_losses_lstm = []

train_time = []
valid_time = []
test_time = []


for i in range(1, 7):
    train_losses_lstm.append(read_file_values('logs/' + str(i) + '/train_loss.txt'))
    train_ppl_lstm.append(read_file_values('logs/' + str(i) + '/train_ppl.txt'))
    valid_losses_lstm.append(read_file_values('logs/' + str(i) + '/valid_loss.txt'))
    valid_ppl_lstm.append(read_file_values('logs/' + str(i) + '/valid_ppl.txt'))

    test_losses_lstm.append(read_file_values('logs/' + str(i) + '/test_loss.txt'))

    train_time.append(read_file_values('logs/' + str(i) + '/train_time.txt'))
    valid_time.append(read_file_values('logs/' + str(i) + '/valid_time.txt'))
    test_time.append(read_file_values('logs/' + str(i) + '/test_time.txt'))


# plot("LSTM train loss by experiment", train_losses_lstm, 'loss', save=SAVE_FIGURES_LSTM, plot=PLOT_FIGURES_LSTM)
# plot("LSTM validation loss by experiment", valid_losses_lstm, 'loss', save=SAVE_FIGURES_LSTM, plot=PLOT_FIGURES_LSTM)
# plot("LSTM train ppl by experiment", train_ppl_lstm, 'ppl', save=SAVE_FIGURES_LSTM, plot=PLOT_FIGURES_LSTM)
# plot("LSTM valid ppl by experiment", valid_ppl_lstm, 'ppl', save=SAVE_FIGURES_LSTM, plot=PLOT_FIGURES_LSTM)

# bar_plot("LSTM test loss by experiment", np.array(test_losses_lstm)[:, 0], 'experiment', 'loss', save=True, plot=True)

train_time = np.array(train_time)
valid_time = np.array(valid_time)
test_time = np.array(test_time)

train_time_exp = train_time.sum(axis=1)
valid_time_exp = valid_time.sum(axis=1)
test_time_exp = test_time[:, 0]

time_by_exp = train_time_exp + valid_time_exp + test_time_exp


# bar_plot('LSTM wall-clock time by experiment', time_by_exp, 'experiment', 'time (s)', save=True, plot=True)


train_losses_vit = []
train_accs_vit = []
train_time_vit = []
valid_losses_vit = []
valid_accs_vit = []
valid_time_vit = []

test_accuracy_vit = []
test_losses_vit = []

test_time_vit = []

for i in range(7, 14):
    train_losses_vit.append(read_file_values('logs/' + str(i) + '/train_loss.txt'))
    train_accs_vit.append(read_file_values('logs/' + str(i) + '/train_accs.txt'))
    train_time_vit.append(read_file_values('logs/' + str(i) + '/train_time.txt'))
    valid_losses_vit.append(read_file_values('logs/' + str(i) + '/valid_loss.txt'))
    valid_accs_vit.append(read_file_values('logs/' + str(i) + '/valid_accs.txt'))
    valid_time_vit.append(read_file_values('logs/' + str(i) + '/valid_time.txt'))

    test_accuracy_vit.append(read_file_values('logs/' + str(i) + '/test_acc.txt'))
    test_losses_vit.append(read_file_values('logs/' + str(i) + '/test_loss.txt'))

    test_time_vit.append(read_file_values('logs/' + str(i) + '/test_time.txt'))


# plot("Vision Transformer train accuracy over epoch by experiment", train_accs_vit, 'Accuracy', save=SAVE_FIGURES_ViT, plot=PLOT_FIGURES_ViT)
# plot_dual("Vision Transformer train accuracy over wall-clock time by experiment", train_accs_vit, train_time_vit, 'Accuracy', save=SAVE_FIGURES_ViT, plot=PLOT_FIGURES_ViT)
# plot("Vision Transformer valid accuracy over epoch by experiment", valid_accs_vit, 'Accuracy', save=SAVE_FIGURES_ViT, plot=PLOT_FIGURES_ViT)
# plot_dual("Vision Transformer valid accuracy over wall-clock by experiment", valid_accs_vit, valid_time_vit, 'Accuracy', save=SAVE_FIGURES_ViT, plot=PLOT_FIGURES_ViT)



bar_plot("ViT test accuracy by experiment", np.array(test_accuracy_vit)[:, 0], 'experiment', 'accuracy', save=True, plot=True)
bar_plot("ViT test loss by experiment", np.array(test_losses_vit)[:, 0], 'experiment', 'loss', save=True, plot=True)

train_time = np.array(train_time_vit)
valid_time = np.array(valid_time_vit)
test_time = np.array(test_time_vit)

train_time_exp = train_time.sum(axis=1)
valid_time_exp = valid_time.sum(axis=1)
test_time_exp = test_time[:, 0]

time_by_exp = train_time_exp + valid_time_exp + test_time_exp


bar_plot('ViT wall-clock time by experiment', time_by_exp, 'experiment', 'time (s)', save=True, plot=True)