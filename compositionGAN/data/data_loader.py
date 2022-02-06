from torch.utils.data import DataLoader
# from data.base_data_loader import BaseDataLoader

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'comp_decomp_unaligned':
        from data.compose_dataset import ComposeDataset
        dataset = ComposeDataset() # unpaired data
    elif opt.dataset_mode == 'comp_decomp_aligned':
        from data.compose_dataset import ComposeAlignedDataset
        dataset = ComposeAlignedDataset() # paired data
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


# class CustomDatasetDataLoader(BaseDataLoader):
class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset_mode = opt.dataset_mode
        self.dataset = CreateDataset(opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, (data) in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data