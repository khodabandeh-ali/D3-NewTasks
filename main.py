import argparse
import numpy as np
import torch

from deepdrugdomain.layers import GraphConvEncoder
from deepdrugdomain.models.DTA.attentiondta_tcbb import AttentionDTA_TCBB
from deepdrugdomain.models.augmentation import AugmentedModelFactory
from deepdrugdomain.optimizers.factory import OptimizerFactory
from deepdrugdomain.schedulers.factory import SchedulerFactory
from torch.utils.data import DataLoader
from deepdrugdomain.models.factory import ModelFactory
from deepdrugdomain.utils.config import args_to_config
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from pathlib import Path
from tqdm import tqdm
import deepdrugdomain as ddd
from torch import nn
import os


def get_args_parser():
    parser = argparse.ArgumentParser(
        'DTIA training and evaluation script', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--raw-data-dir', default='./data/', type=str)
    parser.add_argument('--train-split', default=1, type=float)
    parser.add_argument('--val-split', default=0, type=float)
    parser.add_argument('--dataset', default='drugbank',
                        choices=['dude', 'celegans', 'human', 'drugbank',
                                 'ibm', 'bindingdb', 'kiba', 'davis'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--df-dir', default='./data/', type=str)
    parser.add_argument('--processed-file-dir',
                        default='./data/processed/', type=str)
    parser.add_argument('--pdb-dir', default='./data/pdb/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='gpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    config = args_to_config(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    if torch.cuda.is_available() and args.device == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")

    model = ModelFactory.create("fx_ddi")
    model.to(device)
    preprocesses = ddd.data.PreprocessingList(model.default_preprocess(
        "X1", "X2", "Y"))
    dataset = ddd.data.DatasetFactory.create(
        "drugbank_ddi", file_paths="data/drugbank_ddi/", preprocesses=preprocesses)
    datasets = dataset(split_method="random_split",
                       frac=[0.6, 0.2, 0.2], seed=seed)

    collate_fn = model.collate

    data_loader_train = DataLoader(
        datasets[0], batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    data_loader_val = DataLoader(datasets[1], drop_last=False, batch_size=256,
                                 num_workers=8, pin_memory=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(datasets[2], drop_last=False, batch_size=256,
                                  num_workers=8, pin_memory=True, collate_fn=collate_fn)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = OptimizerFactory.create(
        "adamw", model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = SchedulerFactory.create("cosine", optimizer, num_epochs=150, min_lr=1e-5, warmup_epochs=0, warmup_lr=1e-6)
    # scheduler = None
    train_evaluator = ddd.metrics.Evaluator(
        ["accuracy_score", "auc", "f1_score"], threshold=0.5)
    test_evaluator = ddd.metrics.Evaluator(
        ["accuracy_score", "auc", "f1_score", "precision_score"], threshold=0.5)
    epochs = 150
    accum_iter = 1
    print(model.evaluate(data_loader_val, device,
          criterion, evaluator=test_evaluator))
    loss = 1000
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        metrics = model.train_one_epoch(data_loader_train, device, criterion,
                              optimizer, num_epochs=epochs, scheduler=scheduler, evaluator=train_evaluator, grad_accum_steps=accum_iter)
        eval_metrics = model.evaluate(data_loader_val, device,
                             criterion, evaluator=test_evaluator)
        print(eval_metrics)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "last_model.pth")
        if loss > eval_metrics["val_loss"]:
            torch.save(
                {"model": model.state_dict()}, "best_model.pth")
            loss = eval_metrics["val_loss"]
        #
        if scheduler is not None:
            # step LR for next epoch
            scheduler.step(epoch + 1, metrics["loss"])
    print(model.evaluate(data_loader_test, device,
                         criterion, evaluator=test_evaluator))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DTIA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
