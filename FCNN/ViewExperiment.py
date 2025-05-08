#Currently is hardcoded to follow the structure of similarity type deficit

import matplotlib.pyplot as plt
import numpy as np

def ViewExperiment(deficit, deficit_duration, subset_size):
    subset_size_string = str(subset_size).replace('.', '-')

    save_dir = f'{deficit}/epochs_{deficit_duration}_size_{subset_size_string}/'

    train_loss_list = np.load(save_dir + 'train_losses.npy')
    train_acc_list = np.load(save_dir + 'train_accuracies.npy')
    test_loss_list = np.load(save_dir + 'test_losses.npy')
    test_acc_list = np.load(save_dir + 'test_accuracies.npy')

    num_epochs = train_loss_list.shape[0]
    
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_loss_list, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot accuracy curves
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_acc_list, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Print final test accuracy
    print(f"Final Test Accuracy: {test_acc_list[-1]:.2f}%")