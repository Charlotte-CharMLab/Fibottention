import matplotlib.pyplot as plt
import json
import torch
import numpy as np

# # List to store accuracy data from each file
all_test_acc = []

# File paths to your text files
file_paths = ['../baseline_100/log.txt', '../kqdde_40_100/log.txt' ,'../kqdde_60_100/log.txt','../kqdde_80_100/log.txt']
file_names = ["Baseline", "kqdde_40","kqdde_60","kqdde_80"]

# Loop through each file
for file_path in file_paths:
    epochs = []
    test_acc1 = []
    # Read the file and extract data
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Check if the line is not empty
                json_data = json.loads(line)
                epochs.append(json_data['epoch'])
                test_acc1.append(json_data['test_acc1'])

    # Append the test accuracy values to the global array
    all_test_acc.append(test_acc1)

# Convert the list of arrays to a NumPy array
all_test_acc_np = np.array(all_test_acc)

# # Calculate mean and standard deviation across files
mean_acc = all_test_acc_np.mean(axis=0)
std_acc = all_test_acc_np.std(axis=0)

# Plotting the graph with different line styles for each file
fig, ax = plt.subplots(1)
for i in range(len(file_names)):
    ax.plot(epochs, all_test_acc_np[i], label = file_names[i],linestyle='-')
    
#ax.plot(epochs, mean_acc, lw=2, label='Mean Test Accuracy', color='black', linestyle='--')
# ax.fill_between(epochs, all_test_acc_np[0], all_test_acc_np[1], color='C0', alpha=0.4 , label = "between base & 40%")
# ax.fill_between(epochs, all_test_acc_np[1], all_test_acc_np[2], color='C1', alpha=0.4 , label = "between 40% & 60%")
# ax.fill_between(epochs, all_test_acc_np[2], all_test_acc_np[3], color='C2', alpha=0.4 , label = "between 60% & 80%")
ax.set_title('CIFAR-100 - Random Masking ' , fontsize = 15)
ax.legend(loc='lower right')
ax.set_xlabel('Epochs', fontsize = 15)
ax.set_ylabel('Test Accuracy' , fontsize = 15)
ax.grid()

# Save the graph as an image file (e.g., PNG)
plt.savefig('cifar_100_kqdde.png')

# Show the plot
plt.show()

# # List to store accuracy data from each file
all_test_acc = []

# File paths to your text files
file_paths = ['../baseline_10/log.txt', '../log_kqdde40.txt', '../log_kqdde60.txt','../log_kqdde80.txt',]
file_names = ["Baseline", "kqdde40", "kqdde60",'kqdde80']

# Loop through each file
for file_path in file_paths:
    epochs = []
    test_acc1 = []
    # Read the file and extract data
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Check if the line is not empty
                json_data = json.loads(line)
                epochs.append(json_data['epoch'])
                test_acc1.append(json_data['test_acc1'])

    # Append the test accuracy values to the global array
    all_test_acc.append(test_acc1)

# Convert the list of arrays to a NumPy array
all_test_acc_np = np.array(all_test_acc)

# # # Calculate mean and standard deviation across files
# mean_acc = all_test_acc_np.mean(axis=0)
# std_acc = all_test_acc_np.std(axis=0)

# Plotting the graph with different line styles for each file
fig, ax = plt.subplots(1)
for i in range(len(file_names)):
    ax.plot(epochs, all_test_acc_np[i], label = file_names[i],linestyle='-')
    
#ax.plot(epochs, mean_acc, lw=2, label='Mean Test Accuracy', color='black', linestyle='--')
# ax.fill_between(epochs, all_test_acc_np[0], all_test_acc_np[1], color='C0', alpha=0.4 , label = "between base & 40%")
# ax.fill_between(epochs, all_test_acc_np[1], all_test_acc_np[2], color='C1', alpha=0.4 , label = "between 40% & 60%")
# ax.fill_between(epochs, all_test_acc_np[2], all_test_acc_np[3], color='C2', alpha=0.4 , label = "between 60% & 80%")
ax.set_title('CIFAR-10 With Random Masking', fontsize = 15)
ax.legend(loc='lower right')
ax.set_xlabel('Epochs', fontsize = 15)
ax.set_ylabel('Test Accuracy' , fontsize = 15)
ax.grid()

# Save the graph as an image file (e.g., PNG)
plt.savefig('cifar_10_kqdde.png')

# Show the plot
plt.show()

# # Plotting the graph
# fig, ax = plt.subplots(1)
# ax.plot(epochs, mean_acc, lw=2, label='Mean Test Accuracy')
# ax.plot(epochs, all_test_acc_np[0], lw=2, label='Test Accuracy')
# ax.fill_between(epochs, 0, mean_acc, where=(mean_acc > 0), color='C0', alpha=0.4)
# ax.fill_between(epochs, 0, all_test_acc_np[0], where=(all_test_acc_np[0] > 0), color='C1', alpha=0.4)
# ax.set_title('Test Accuracy with $\pm \sigma$ Interval Across Files')
# ax.legend(loc='upper left')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Test Accuracy')
# ax.grid()

# # Save the graph as an image file (e.g., PNG)
# plt.savefig('test_accuracy_plot.png')

# # Show the plot
# plt.show()



# Create a sample PyTorch tensor with size 197x64
# Create a sample PyTorch tensor with size 197x64 and values between 0 and 255

# import torch
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# # Generate twelve sample PyTorch tensors with size (197, 64) and values between 0 and 255
# num_tensors = 12
# tensors = [torch.randint(low=0, high=256, size=(197, 64), dtype=torch.float32) for _ in range(num_tensors)]

# # Concatenate the tensors along a new dimension (e.g., create a 3D tensor)
# combined_data = torch.stack(tensors, dim=0)

# # Flatten the combined tensor along the batch dimension
# flattened_data = combined_data.view(-1, 64)

# # Perform t-SNE to reduce dimensionality to 2D
# tsne = TSNE(n_components=2)
# data_tsne = tsne.fit_transform(flattened_data)

# # Plot the 2D t-SNE representation for each tensor
# colors = plt.cm.rainbow([i / float(num_tensors) for i in range(num_tensors)])
# for i in range(num_tensors):
#     start_index = i * combined_data.size(1)
#     end_index = (i + 1) * combined_data.size(1)
#     plt.scatter(data_tsne[start_index:end_index, 0], data_tsne[start_index:end_index, 1], label=f'Tensor {i}', color=colors[i])

# plt.title('t-SNE Visualization of PyTorch Tensors')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.legend()
# plt.show()






