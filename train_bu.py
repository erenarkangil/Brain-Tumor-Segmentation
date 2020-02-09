import torch
import torchvision
import torch.optim
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

import sys
import os
import numpy as np
from time import time, gmtime, strftime
from tqdm import tqdm

import nibabel as nib
from scipy.ndimage import zoom

import wnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################################
# Set params
########################################
if (len(sys.argv) > 2):
    root_path = sys.argv[3]
else:
    root_path = "."

loss_to_take = 'ce'  # 'dice' or 'l2' or 'ce'
n_epochs = 100
batch_size = 16
b_shuffle = True
n_workers = 4  # parallel processing. enable by setting to some natural number
########################################
# DataSet Class
# 
# This class must implement __init__(self), __len__(self) and __getitem(self, idx) functions.
# The pattern is used to reading and pre-processing the data.
# Pre-processing should preferrably be done via transforms, which in turn should only use
# torch transformations (for better runtime).
# Input shape name can be misleading: what is meant is the size of each volume. Since we have 4 modalities,
# the actual input of the network will then be (4, *input_shape).
# load_set = 3 -> training set loaded, load_set = 0 validation set loaded, load_set = 1 -> test set loaded
########################################

print("Batch_size {} device {} workers {} loss {}".format(batch_size, device, n_workers, loss_to_take))


class BRATS(torch.utils.data.Dataset):
    def __init__(self, path_to_root, transform=None, input_shape=(80, 96, 64), load_set=1):
        self.root = path_to_root
        self.samples = []
        self.transform = transform
        self.input_shape = (4, *input_shape)
        self.output_shape = (3, *input_shape)
        self.read_dataset(load_set)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        data = np.empty(self.input_shape, dtype=np.float32)
        labels = np.empty(self.output_shape, dtype=np.float32)
        types = [f for f in os.listdir(self.samples[idx]) if os.path.isfile(os.path.join(self.samples[idx], f))]

        for f in types:
            f = os.path.join(self.samples[idx], f)
            img = nib.load(f).get_fdata()

            # we need to reshape, so calculate the factors
            # by the way: normally we would do spatial normalization as well
            # but BRATS is already normalized in that regard!
            factors = (
                labels.shape[1]/img.shape[0],
                labels.shape[2]/img.shape[1], 
                labels.shape[3]/img.shape[2]
            )

            # labelmaps and actual image data is processed differently
            if f.endswith('_seg.nii.gz'):
                # reshape first to input size of model
                img = zoom(img, factors, order=0)
                # to one hot
                labels = np.zeros(self.output_shape, np.float32)  # 3 because 3 labels are of relevance for our training
                for lb in [1, 2, 4]:  # iterate through label classes
                    if lb == 4:
                        y = 2
                    else:
                        y = lb - 1
                    labels[y, img == lb] = 1

            else:
                # reshape the data to inputsize of model
                img = zoom(img, factors, order=3)
                # normalize volume
                mean = img.mean()
                std = img.std()
                img = (img - mean) / std

                if f.endswith('_t1.nii.gz'):
                    data[0, :, :, :] = img
                elif f.endswith('_t1ce.nii.gz'):
                    data[1, :, :, :] = img
                elif f.endswith('_2.nii.gz'):
                    data[2, :, :, :] = img
                elif f.endswith('_flair.nii.gz'):
                    data[3, :, :, :] = img

        assert not np.any(np.isnan(data)), self.samples[idx]
        assert not np.any(np.isnan(labels)), self.samples[idx]
        # to torch tensor for faster processing
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

        # # if cuda is available, use it
        # if torch.cuda.is_available():
        #     device = torch.device("cuda")
        #     data = data.to(device)
        #     labels = labels.to(device)


        if self.transform:  # Transforms are applied to the concatenated tensor of all modalities
            data = self.transform(data)

        return (data, labels) # sample


    '''
    Reads all files 
    '''
    def read_dataset(self, load_set):

        if load_set == 3:
            # find test and validation files
            exclude_files = ''
            with open('VAL_TEST.csv', 'r') as read_file:
                exclude_files = read_file.read()

            exclude_files = exclude_files.split('\n')
            exclude_files = exclude_files[0][4:] + ',' + exclude_files[1][4:]
            exclude_files = exclude_files.split(',')

            # create dataset
            for rt, dirs, files in os.walk(os.path.join(self.root, 'DATA', 'HGG')):
                for name in dirs:
                    name = os.path.join(self.root, 'DATA', 'HGG', name)
                    if name not in exclude_files:
                        self.samples.append(name)

            for rt, dirs, files in os.walk(os.path.join(self.root, 'DATA', 'LGG')):
                for name in dirs:
                    name = os.path.join(self.root, 'DATA', 'LGG', name)
                    if name not in exclude_files:
                        self.samples.append(name)

        elif load_set == 0 or load_set == 1:

            with open('VAL_TEST.csv', 'r') as read_file:
                data = read_file.read()

                self.samples = data.split('\n')[load_set][4:].split(',')

        print("This dataset has {:4} items".format(len(self.samples)))

# if __name__ == '__main__':
    # dataset = NumbersDataset()
    # print(len(dataset))
    # print(dataset[100])
    # print(dataset[122:361])

########################################
# Reading, preprocessing and batching pipeline
########################################

# First create transforms
pre_processing = torchvision.transforms.Compose([
#                                                 torchvision.transforms.ToTensor(),  # creates torch tensor from the data -> faster processing
                                                    ])

# Second create Dataset object
train_data = BRATS(root_path, transform=pre_processing, load_set=3)
val_data = BRATS(root_path, transform=pre_processing, load_set=0)

# Third create Dataloader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=b_shuffle, num_workers=n_workers)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                           shuffle=b_shuffle, num_workers=n_workers)

########################################
# Initialize model and optimizer and loss and writer
########################################

model = wnet.VNet()
model.to(device)


optimizer = torch.optim.Adam(model.parameters())
# at this point, we could use a learning rate scheduler. But for now, we don't use it

writer = SummaryWriter(os.path.join(root_path, "training", strftime("%Y%m%d_%H%M%S", gmtime())))

class Dice(torch.nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, pred, target):
        numerator = 2 * torch.sum(pred * target)
        denominator = torch.sum(pred + target)
        return 1 - (numerator + 1) / (denominator + 1)

if loss_to_take == 'dice':
    loss_func = Dice()
elif loss_to_take == 'l2':
    loss_func = torch.nn.MSELoss()
elif loss_to_take == 'ce':
    loss_func = torch.nn.BCELoss()
else:
    raise ValueError('Invalid arugment for variable loss_to_take: {}'.format(loss_to_take))

########################################
# define training routine
########################################

def train(i_epoch = 0):

    model.train()
    
    iteration = len(train_loader) * i_epoch

    for x, y_hat in train_loader:


        # if cuda is available, use it
        x = x.to(device)
        y_hat = y_hat.to(device)

        optimizer.zero_grad()

        y = model(x)

        y = y.flatten()
        y_hat = y_hat.flatten()

        loss = loss_func(y, y_hat)            # crossentropy loss. If one class is too imbalanced,
                                              # we can use the weights argument!
        loss.backward()

        optimizer.step()

        if (iteration % 1 == 0):
            print("Iteration {:10} Loss {:10}".format(iteration, loss.data.item()))

        if (iteration % 10 == 0):
            writer.add_scalar("loss", loss.data.item(), iteration)

            # img = 
            # writer.add_image("trainimg", y)

        iteration += 1
        

def validate(i_epoch, best_loss):

        loss_val = 0

        model.eval()

        for x, y_hat in val_loader:

            x = x.to(device)
            y_hat = y_hat.to(device)

            y = model(x)
            # if (loss_val == 0):
            #     writer.add_image("val image", toTensor(img), iteration)

            y = y.flatten()
            y_hat = y_hat.flatten()

            loss_val += loss_func(y, y_hat)  # crossentropy loss. If one class is too imbalanced,
                                                  # we can use the weights argument!
            
        loss_val /= len(val_loader)


        writer.add_scalar("val", loss_val.data.item(), i_epoch)

        print("Validation loss: {:10}".format(loss_val.data.item()))

        if (loss_val < best_loss):
            if not os.path.exists('models'):
                os.makedirs('models')
        torch.save({'state_dict': model.state_dict()}, os.path.join("models", "mdl" + str(i_epoch) + ".pth.tar"))
        
        return loss_val
########################################
# do training
########################################

if (__name__ == '__main__'):
    try:
        start = time()
        best_loss = float("inf")
        for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):
            print("")

            train(i_epoch)

            with torch.no_grad():
                last_loss = validate(i_epoch, best_loss)

                if (last_loss < best_loss):
                    best_loss = last_loss

            print("Finnished epoch {:4}".format(i_epoch))

    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        print(f"\n\nTraining took {(time()-start)/60:.2f} minutes\n")


###########################################################################
# PLAYGROUND
###########################################################################
# 1.
# To visualize what's going on in prveious classes play with:
# num = int(_data.__len__()/2)
# example, ex_labels = train_data.__getitem__(num);

# batch = next(iter(train_loader))
# print(batch[0].shape, batch[1].shape)
