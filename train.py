import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from dataset import IMDBDataset
from model import Model
from tqdm import tqdm


class Train:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        self.train_dataset = IMDBDataset(self.train_path)
        self.test_dataset = IMDBDataset(self.test_path)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.net = Model().to(self.device)
        self.model_path = 'models/net.pth'

        if os.path.exists(self.model_path):
            self.net.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            print('models load successfully.')
        else:
            print('no models.')

        self.batch_size = 256
        self.num_workers = 6

    def train(self, epochs=500, best_acc=0.5):
        train_dataloder = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers)

        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters())

        for i in range(epochs):
            print(f"epochs: {i}")

            self.net.train()
            for j, (data, target) in enumerate(train_dataloder):
                data = data.to(self.device)
                target = target.to(self.device)

                out = self.net(data)

                loss = loss_fn(out, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"epochs:{i} -> {j}/{len(train_dataloder)} loss: {loss.item()}")

            acc = self.test()

            if acc > best_acc:
                best_acc = acc
                torch.save(self.net.state_dict(), self.model_path)
                print('save models.')

            print('best accuracy: {}'.format(best_acc))

    def test(self):
        print('test datasets calculate accuracy...')

        test_dataloder = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers)

        correct = 0
        total = 0
        with torch.no_grad():
            self.net.eval()
            for data, target in tqdm(test_dataloder):
                data = data.to(self.device)
                out = self.net(data).cpu()

                out = out.reshape(-1)
                target = target.reshape(-1)

                correct += torch.sum(torch.eq(out.round(), target.round())).item()
                total += out.shape[0]

        acc = correct / total
        print(f'accuracy: {acc}')

        return acc


if __name__ == '__main__':
    t = Train('data/train.txt', 'data/test.txt')
    # t.train()
    t.test()
