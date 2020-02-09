import sys
import torch
import numpy as np
import nibabel as nib
import wnet
import os
from scipy.ndimage import zoom

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
                img = zoom(img, factors, order=2)
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

def save_tensors(tensors, save_path):

    shapetens = tensors[0].size()
    dimensions = (shapetens[2], shapetens[3], shapetens[4])

    arr = np.zeros(dimensions)


    endings = ['input', 'groundtruth', 'predicted']
    i = 0

    while i < len(tensors):
        
        ten = tensors[i].detach().numpy()

        # labels from ont_hot to labels
        if i == 1:
            arr = np.zeros(dimensions)

            for k in range(3):
                sub_tens = ten[0, k, :, :, :]
                arr[sub_tens==1] = k + 1

            ten = arr
            # print("Ten min {} max {}".format(np.amin(ten), np.amax(ten)))

            img = nib.Nifti1Image(ten, np.eye(4))

        # predicted labels, already preprocessed
        elif i == 2:

            # print("predict min {} max {}".format(np.amin(ten), np.amax(ten)))

            img = nib.Nifti1Image(ten, np.eye(4))

        # reshaped input image
        else:

            img = nib.Nifti1Image(ten[0, 1, :, : ,: ], np.eye(4))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filename = os.path.basename(save_path)
        filename = os.path.join(save_path, endings[i] + '.nii.gz')

        nib.save(img, filename)
        i += 1

class Dice(torch.nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, pred, target):
        numerator = 2 * torch.sum(pred * target)
        denominator = torch.sum(pred + target)
        return 1 - (numerator + 1) / (denominator + 1)


class Accuracy(torch.nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, pred, target):

        matches = (pred == target).float().sum()
        return 1.0 * matches / target.numel()

def test_model(model_path, root_path):

    # setup data
    test_data = BRATS(root_path, load_set=1)
    print("There are {:4} files in the test set.".format(len(test_data)))
    test_loader = torch.utils.data.DataLoader(test_data)

    # "loss", we'll use Dice for evaluation as well as Accuracy
    dice = Dice()
    acc = Accuracy()

    d_tot = 0
    a_tot = 0

    # load model
    device = torch.device('cpu')
    model = wnet.VNet()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    # mode.eval()


    i = 0
    dimensions = (80, 96, 64)
    for x, y_hat in test_loader:

        file_name = "Example" + str(i)

        # if cuda is available, use it
        x_in = x.to(device)
        y_hat = y_hat.to(device)
        # predict
        y_pred = model(x_in)
        y_numy = y_pred.detach().numpy()
        arr = np.zeros(dimensions)
        # post process y
        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                for z in range(dimensions[2]):

                    ls = y_numy[0, :, x, y, z]
                    ls = ls[ls > 0.5]

                    if len(ls) == 0:
                        arr[x, y, z] = 0
                    else:
                        arr[x, y, z] = np.argmax(ls) + 1

        # save y, y_hat and x reshaped
        save_tensors([x_in, y_hat, torch.from_numpy(arr)], os.path.join(root_path, "test", file_name))

        # calculate metrics
        y_pred = y_pred.flatten()
        y_hat = y_hat.flatten()

        d = dice(y_pred, y_hat)
        a = acc(y_pred, y_hat)
        d_tot += d
        a_tot += a

        print("Dice {:10} Accuracy {:10}".format(d, a))

        i += 1

    d_tot /= len(test_loader)
    a_tot /= len(test_loader)

    print("Result on complete test set: Dice {:10} Accuracy {:10}".format(d_tot, a_tot))


if (__name__ == '__main__'):

    if (len(sys.argv) != 3):
        assert False, "Use script as follows:\n python3 test_model.py <path_to_your_model> <root_path>"

    test_model(sys.argv[1], sys.argv[2])
