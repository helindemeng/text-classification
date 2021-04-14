import torch
from torch.utils.data import Dataset, DataLoader


class IMDBDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.all_files = f.readlines()

    def __getitem__(self, index):
        filename = self.all_files[index]
        filename_split = filename.strip().split(' ')

        target = int(filename_split[0])

        data = list(map(int, filename_split[1:]))
        data_len = 300
        data = data[:data_len]
        if len(data) < data_len:
            data.extend([1 for _ in range(data_len - len(data))])  # sentence length lower than 300, padding 1

        return torch.tensor(data, dtype=torch.int64), torch.tensor([target], dtype=torch.float32)

    def __len__(self):
        return len(self.all_files)


if __name__ == '__main__':
    for data_, target_ in DataLoader(IMDBDataset('data/train.txt'), batch_size=512, shuffle=True):
        print(data_.shape)
        print(target_.shape)
