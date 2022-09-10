from avalanche.models import SimpleCNN
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.models import as_multitask
import torch.utils.data.dataset
import repackage
repackage.up ()
from torch import nn
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import EWCPlugin
from torch.nn import CrossEntropyLoss
from abc import ABC, abstractmethod
from avalanche.benchmarks.utils import (
    AvalancheDataset,
    AvalancheSubset,
    AvalancheConcatDataset,
)
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
import torch.utils.data.dataset
from torch import optim
from avalanche.training.plugins import ReplayPlugin, EWCPlugin

"""Trains GAN on MNISTs variants"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from mnists.config import get_cfg_defaults
from mnists.dataloader import get_dataloaders
from mnists.models import DiscLin, DiscConv,GenConv
from utils1 import save_cfg, load_cfg,Optimizers

class ExemplarsBuffer(ABC):
    """ABC for rehearsal buffers to store exemplars.

    `self.buffer` is an AvalancheDataset of samples collected from the previous
    experiences. The buffer can be updated by calling `self.update(strategy)`.
    """

    def __init__(self, max_size: int):
        """Init.

        :param max_size: max number of input samples in the replay memory.
        """
        self.max_size = max_size
        """ Maximum size of the buffer. """
        self._buffer: AvalancheDataset = AvalancheConcatDataset([])

    @property
    def buffer(self) -> AvalancheDataset:
        """Buffer of samples."""
        return self._buffer

    @buffer.setter
    def buffer(self, new_buffer: AvalancheDataset):
        self._buffer = new_buffer

    @abstractmethod
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update `self.buffer` using the `strategy` state.

        :param strategy:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def resize(self, strategy: "SupervisedTemplate", new_size: int):
        """Update the maximum size of the buffer.

        :param strategy:
        :param new_size:
        :return:
        """
        ...

class ReservoirSamplingBuffer_gen(ExemplarsBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self,max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super ().__init__(max_size)
        #self.gen_data=gen_data
        # INVARIANT: _buffer_weights is always sorted.
        #self._buffer_weights = torch.zeros(0)
        self.buffer_gen=None
        self._buffer_weights_gen = None
    def update(self, strategy: "Cumulative", **kwargs):
        """Update buffer."""
        self.update_from_dataset_gen(strategy.experience.dataset,strategy.gen_data)

    def update_from_dataset_gen(self, new_data: AvalancheDataset,gen_data:AvalancheDataset):
        """Update the buffer using the given dataset.
        :param new_data:
        :return:
        """
        new_weights = torch.rand(len(new_data))
        gen_weights = torch.rand(len(gen_data))


        cat_weights = torch.cat([new_weights, gen_weights])
        cat_data = AvalancheConcatDataset([new_data, gen_data])
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer_gen = AvalancheSubset(cat_data, buffer_idxs)
        self._buffer_weights_gen = sorted_weights[: self.max_size]
        #strategy.get_buff(self.buffer)
    def resize(self,strategy,  new_size):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer_gen) <= self.max_size:
            return
        self.buffer = AvalancheSubset(self.buffer_gen, torch.arange(self.max_size))
        self._buffer_weights_gen = self._buffer_weights_gen[: self.max_size]


class ReplayP_conterfact(SupervisedPlugin):

    def __init__(self, mem_size):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        #self.gen_data = data
        self.buffer = ReservoirSamplingBuffer_gen(max_size=mem_size)

    def before_training_exp(self, strategy: "Cumulative",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Use a custom dataloader to combine samples from the current data and memory buffer. """
        if len(self.buffer.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.buffer.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "Cumulative", **kwargs):
        """ Update the buffer. """
        self.buffer.update(strategy, **kwargs)


from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.templates import SupervisedTemplate


class Cumulative(SupervisedTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
          # cumulative dataset
        self.gen_data=None
    def get_data(self,data):
        self.gen_data=data
        return self.gen_data

    def get_buff(self,a):
        self.gen_data=a



def save(x, path, n_row, sz=64):
    x = F.interpolate(x, (sz, sz)) #采样函数，采样成（sz,sz）
    save_image(x.data, path, nrow=n_row, normalize=True, padding=2)


def sample_image(generator, sample_path, batches_done, device, n_row=3, n_classes=10):
    """Saves a grid of generated digits"""
    y_gen = np.arange(n_classes).repeat(n_row)
    y_gen = torch.LongTensor(y_gen).to(device)
    x_gen = generator(y_gen)

    save(x_gen.data, f"{sample_path}/0_{batches_done:d}_x_gen.png", n_row)


def fit(cfg, generator, discriminator, x,y, opts, losses, device):
    # directories for experiments
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_path = Path('.')/'mnists'/'experiments'
    model_path /= f'gan_{cfg.TRAIN.DATASET}_{time_str}_{cfg.MODEL_NAME}'
    weights_path = model_path/'weights'
    sample_path = model_path/'samples'
    weights_path.mkdir(parents=True, exist_ok=True)
    sample_path.mkdir(parents=True, exist_ok=True)

    # dump config
    save_cfg(cfg, model_path/"cfg.yaml")

    # Training Loop 将数据转成8个一组
    L_adv= losses
    x_=len(x)
    y_=len(y)
    x_=x_//8
    y_=y_//8
    x1=x[0:x_*8]
    y1=y[0:y_*8]
    x=x1.reshape(x_,8,3,32,32)[1:]
    y=y1.reshape(y_,8)[:]
    for i, data in enumerate(x):
        # Data and adversarial ground truths to device
        x_gt = data.to(device)
        y_gt = y[i].to(device)
        valid = torch.ones(len(y_gt), ).to(device)
        fake = torch.zeros(len(y_gt),).to(device)

        #
        #  Train Generator
        #
        opts.zero_grad (['generator'])

        # Sample noise and labels as generator input
        y_gen = torch.randint(cfg.MODEL.N_CLASSES, (len( y_gt), )).to(device)

        # Generate a batch of images
        x_gen = generator(y_gen)

        # Calc Losses
        validity = discriminator(x_gen, y_gen)

        losses_g = {}
        losses_g['adv'] = L_adv(validity, valid)

        # Backprop and step
        loss_g = sum(losses_g.values())
        loss_g.backward ()
        opts.step(['generator'], False)

        #
        # Train Discriminator
        #
        opts.zero_grad(['discriminator'])

        # Discriminate real and fake
        validity_real = discriminator(x_gt, y_gt)
        validity_fake = discriminator(x_gen.detach (), y_gen)

        # Losses
        losses_d = {}
        losses_d['real'] = L_adv(validity_real, valid)
        losses_d['fake'] = L_adv(validity_fake, fake)
        loss_d = sum(losses_d.values())/2

        # Backprop and step
        loss_d.backward ()
        opts.step ( ['discriminator'], False )

    return generator

def merge_args_and_cfg(args, cfg):
    cfg.MODEL_NAME = args.model_name
    cfg.LOG.SAVE_ITER = args.save_iter
    cfg.TRAIN.EPOCHS = args.epochs
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    return cfg



def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='',
                          help="path to a cfg file" )
    parser.add_argument('--model_name', default='tmp',
                          help='Weights and samples will be saved under experiments/model_name')
    parser.add_argument("--save_iter", type=int, default=10000,
                          help="interval between image sampling" )
    parser.add_argument("--epochs", type=int, default=15,
                          help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8,
                          help="size of the batches")
    args = parser.parse_args()

    # get cfg
    cfg = load_cfg(args.cfg) if args.cfg else get_cfg_defaults()
    cfg = merge_args_and_cfg(args, cfg )
    Generator = GenConv
    generator_cgn = Generator(
        n_classes=cfg.MODEL.N_CLASSES,
        latent_sz=cfg.MODEL.LATENT_SZ,
        ngf=cfg.MODEL.NGF,
    )

    Discriminator = DiscLin if cfg.MODEL.DISC == 'linear' else DiscConv
    discriminator = Discriminator(n_classes=cfg.MODEL.N_CLASSES, ndf=cfg.MODEL.NDF)
    L_adv = torch.nn.BCEWithLogitsLoss()


    opts = Optimizers()
    opts.set('generator', generator_cgn, lr=cfg.LR.LR, betas=cfg.LR.BETAS)
    opts.set('discriminator', discriminator, lr=cfg.LR.LR, betas=cfg.LR.BETAS)

    # push to device and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_cgn = generator_cgn.to(device)
    discriminator = discriminator.to(device)
    losses = L_adv.to ( device )
    return cfg, generator_cgn, discriminator, opts, losses, device




class simpleCNN_T(SimpleCNN):
    def __init__(self,num_classes=10):
        super(simpleCNN_T, self).__init__()
        self.num_classes=num_classes
        self.classifier = nn.Sequential(
            nn.Linear(64, self.num_classes),
            )
        self.out=nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.out(x)
        return x



# data :color mnist包装成AvalancheDataset
dl_train, dl_test = get_dataloaders()
train_val = []
train_lab = []
for i, data in enumerate(dl_train):
    k = data['ims'].tolist()
    j = data['labels'].tolist()
    for ii in range(8):
        train_val.append(k[ii])
        train_lab.append(j[ii])
dl_train_val = train_val
dl_train_lab = train_lab
a = torch.tensor(dl_train_val)
b = torch.tensor(dl_train_lab)
dl_train = AvalancheDataset(TensorDataset(a, b))

test_val = []
test_lab = []
for i, data in enumerate(dl_test):
    k = data['ims'].tolist()
    j = data['labels'].tolist()
    for ii in range(8):
        test_val.append(k[ii])
        test_lab.append(j[ii])
dl_test_val = test_val
dl_test_lab = test_lab
a = torch.tensor(dl_test_val)
b = torch.tensor(dl_test_lab)
dl_test = AvalancheDataset(TensorDataset(a, b))

scenario = nc_benchmark(
    dl_train, dl_test, n_experiences=5, shuffle=True, seed=1234,
    task_labels=False
)

train_stream = scenario.train_stream
test_stream = scenario.test_stream


# model
cfg, generator_cgn, discriminator, opts, losses_cgn, device = main1()
generator_cgn.train()
discriminator.train()


device = torch.device( "cuda:0" if torch.cuda.is_available () else "cpu" )
model=simpleCNN_T(num_classes=2)
model = as_multitask(model, 'classifier')

optimizer = optim.Adam(model.parameters (), lr=0.008)
criterion = CrossEntropyLoss().to(device)
replay = ReplayP_conterfact(mem_size=200)
replay1 = ReplayPlugin(mem_size=100)
ewc = EWCPlugin(ewc_lambda=0.001)
strategy = Cumulative(
    model, optimizer, criterion,train_mb_size=100, train_epochs=10, eval_mb_size=100, device=device,
    plugins=[replay,ewc])


# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train_gan 网络
    x=torch.Tensor(1,3,32,32)
    y=torch.tensor([0])
    for k in [experience]:
        for i in range(len(k.dataset)):
            k_1=k.dataset[i][0].to(torch.device("cpu"))
            k_2 = torch.tensor([k.dataset[i][1]]).to(torch.device("cpu"))
            x = torch.cat([x,k_1.reshape(1,3,32,32)])
            y = torch.cat([y, k_2])
    model_cgn = fit(cfg, generator_cgn, discriminator, x, y, opts, losses_cgn, device)
    lis_tensor_label = y


    # generrater data :size(1000)
    gen_x = torch.Tensor(1, 3, 32, 32).to(device)
    gen_y = torch.tensor([0]).to(device)
    for i in range(200):
        data = torch.tensor([y[i]])
        data = model_cgn(data.to(device)).detach()
        gen_x = gen_x.to(device)
        print('-------------0----------',gen_x.shape)
        gen_x = torch.cat([gen_x, data], 0).to(device)
        data_y = torch.tensor([lis_tensor_label[i]]).to(device)
        gen_y = torch.cat([gen_y, data_y])
        print(gen_y.shape)

    gen_x = gen_x.to(torch.device("cpu"))[1:]
    gen_y = gen_y.to(torch.device("cpu"))[1:]
    print(gen_x.shape)
    print(gen_y.shape)
    gen_train = AvalancheDataset(TensorDataset(gen_x, gen_y))

    # train continue-learn simpleCNN
    strategy.get_data(gen_train)
    strategy.train ( experience )
torch.save(model.state_dict(), "mnists/models_clearn_mt_simplcnn39_overfit.pth")
