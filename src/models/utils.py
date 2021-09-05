import torch


def train(model, data_loader, optimizer, criterion, device=None):

    # Setup train and device
    device = device or torch.device("cpu")
    model.train()

    # Metrics
    running_loss = 0.0
    epoch_steps = 0

    for data, target in data_loader:

        # get the inputs; data is a list of [inputs, labels]
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_steps += 1

    return running_loss, epoch_steps


def test(model, data_loader, criterion, device=None):
    # Setup eval and device
    device = device or torch.device("cpu")
    model.eval()
    predicted = []
    targets = []
    losses = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            # Append data
            targets.append(target)
            predicted.append(outputs)
            losses.append(loss)

    return predicted, targets, losses
