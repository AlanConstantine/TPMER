import torch
import torch.nn as nn
import torch.optim as optim
from tools import *
from config import Params
from tqdm import tqdm
import sys

from representation.PhySiRES import MultiSignalRepresentation
from models.MER import MERClassifer


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(
            1, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(
            1, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# define the training function


def train(model, optimizer, criterion, train_loader, test_loader, n_epochs):
    for epoch in range(n_epochs):
        print('Epoch [%d/%d]' % (epoch+1, n_epochs))
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader),
                    total=len(train_loader), file=sys.stdout)
        for i, batch in enumerate(loop):
            # print(batch[0])
            inputs, labels = batch[1][0], batch[1][1]
            labels = labels.flatten()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step_loss = {'train_loss': loss.item()}

            if i != len(train_loader) - 1:
                loop.set_postfix(**step_loss)
            else:
                epoch_loss = running_loss / i
                epoch_log = {'train_loss': epoch_loss}

                loop.set_postfix(**epoch_log)

        test(model, criterion, test_loader)


def test(model, criterion, test_loader):
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in test_loader:
            labels = labels.flatten()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        accuracy = 100.0 * total_correct / total_samples
        print('Test Accuracy: %.2f%% (%d/%d)\n' %
              (accuracy, total_correct, total_samples))


def main():
    args = Params(use_cuda=True, debug=False, data=r'./processed_signal/HKU956/956_772_12s_step_4s.pkl', batch_size=8, target='4d',
                  spliter=r'./processed_signal/HKU956/956_772_12s_step_4s_spliter5.pkl',)

    spliter = load_model(args.spliter)
    data = pd.read_pickle(args.data)

    for i, k in enumerate(spliter[args.valid]):
        args.k = i
        print("\n" + "=======" * 6 + '[Fold {}]'.format(i), "=======" * 6)
        train_index = k['train_index']
        test_index = k['test_index']
        break

    dataprepare = DataPrepare(args,
                              target=args.target,
                              data=data,
                              train_index=train_index,
                              test_index=test_index,
                              device=args.device,
                              batch_size=args.batch_size)
    train_dataloader, test_dataloader = dataprepare.get_data()

    input_dim = 4
    hidden_dim = 32
    output_dim = 4
    learning_rate = 0.0001
    n_epochs = 100

    # create the model
    model = MultiSignalRepresentation(
        output_size=40, seq=768, device=args.device, )
    model.output_layer = MERClassifer(args, n_class=2)
    # model = LSTMClassifier(input_dim, hidden_dim,
    #                        output_dim, device=args.device)
    model = model.to(args.device)

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # create the dataset and dataloader
    train_data = torch.randn(1000, input_dim, 10).to(args.device)
    train_labels = torch.randint(0, output_dim, (1000,)).to(args.device)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)

    test_data = torch.randn(50, input_dim, 10).to(args.device)
    test_labels = torch.randint(0, output_dim, (50,)).to(args.device)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False)

    # train(model, optimizer, criterion,
    #       train_loader, test_loader, n_epochs)

    train(model, optimizer, criterion,
          train_dataloader, test_dataloader, n_epochs)


if __name__ == "__main__":
    main()
