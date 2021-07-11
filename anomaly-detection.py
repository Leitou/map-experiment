
# select some hyperparameters and initialize network
learning_rate = 1e-5
epochs = 100
m = 0.9

net = NeuralNetwork(X_train.shape[1]).to(device)
#print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=m)
loss_fn = nn.BCEWithLogitsLoss()


# 4 Train the network using categorical cross-entropy and SoftMax
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch %  30 == 0:
            loss, current = loss.item(), batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    model.eval() # switch to testing mode, if theres a difference in training and testing, e.g. dropout,
    with torch.no_grad(): # tells that no need to build a derivative graph, performance
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred[pred < 0.5] = 0
            pred[pred > 0.5] = 1

            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

train = True

if train:
    acc = 0
    for e in range(epochs):
        print(f"Epoch {e + 1}: run train and test loop")
        train_loop(train_loader, net, loss_fn, optimizer)
        tacc = test_loop(test_loader, net, loss_fn)
        if tacc > acc:
            acc = tacc
            torch.save(net.state_dict(), "upsampling-mlp.model")
    print(f"Done! highest accuracy ever achieved is: {acc}")
else:
    net.load_state_dict(torch.load("upsampling-mlp.model"))
    tacc = test_loop(test_loader, net, loss_fn)
    print(f"Done! highest accuracy ever achieved is: {tacc}")