import torch


def calculate_accuracy(model, data_iterator, key=None):
    """Calculate the classification accuracy.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : dict
        A dictionary containing data iterator object.
    key : str
        A key to select which dataset to evaluate

    Returns
    -------
    float
        accuracy of classification for the given key.

    """
    if key is None:
        keys = data_iterator.keys()
    else:
        keys = [key]  # make it as a list

    accuracy = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for key in keys:
            total = 0
            length = 0

            # Set model in evaluation mode
            model.eval()

            for x, y in data_iterator[key]:
                out_put = model(x.to(device))
                out_put = out_put.cpu().detach()
                total += (out_put.argmax(dim=1) == y.argmax(
                    dim=1)).float().sum()
                length += len(y)
            accuracy[key] = (total / length).numpy()
    return accuracy
