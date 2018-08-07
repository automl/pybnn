def infinite_dataloader(dataloader):
    """ Yield an unbounded amount of batches from a `torch.utils.data.DataLoader`.
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Iterable yielding batches of data from a dataset of interest.
    """
    while True:
        for batch in dataloader:
            yield batch