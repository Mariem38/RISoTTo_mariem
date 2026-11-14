import csv
import json
import os
import warnings
import sys

from tqdm import tqdm
import h5py

from config import config_data, config_model, config_runtime
from data_handler import Dataset
from model import Model
import numpy as np
from src.data_encoding import extract_topology, std_rna
from src.dataset import select_by_max_ba, select_by_sid
from src.logger import Logger
from src.scoring import bc_score_names, bc_scoring, nanmean
import torch as pt
from torch.utils.data import DataLoader, Sampler
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

class GroupSampler(Sampler):

    def __init__(self, dataset, samples_filepath, p_mod=0.5, epoch_size=None, weight_exp=0.5):
        info = {}
        with open(samples_filepath, newline='') as f:
            for r in csv.DictReader(f):
                info[r['sample'].upper()] = (r['group'], any(int(r[c]) > 0 for c in ('n_PSU', 'n_5MC', 'n_OMG')))
        # sample ids of the selected keys, as in select_by_sid
        sids = ['_'.join([s.split(':')[0] for s in key.split('/')[1::2]]).upper() for key in dataset.keys[dataset.m]]
        self.pools = {True: {}, False: {}}
        for i, sid in enumerate(sids):
            g, mod = info.get(sid, (f"missing:{sid}", False))
            self.pools[mod].setdefault(g, []).append(i)
        self.pools = {k: list(v.values()) for k, v in self.pools.items() if v}
        self.p_mod = p_mod if (True in self.pools and False in self.pools) else float(True in self.pools)

        # group weights: probability of a group within its pool ∝ size**weight_exp
        # (0 = every group equally, 0.5 = sqrt(size), 1 = every sample equally)
        self.probs = {}
        for k, groups in self.pools.items():
            w = np.array([len(g) for g in groups], dtype=float) ** weight_exp
            self.probs[k] = w / w.sum()

        # cycling: each group goes through a shuffled order of its members before repeating one;
        # the position is kept across epochs (the sampler lives as long as the DataLoader)
        self.rng = np.random.default_rng()
        self.order = {k: [None] * len(groups) for k, groups in self.pools.items()}
        self.pos = {k: [0] * len(groups) for k, groups in self.pools.items()}

        n_groups = len({g for g, _ in (info.get(s, (f"missing:{s}", False)) for s in sids)})
        self.epoch_size = int(epoch_size) if epoch_size else n_groups
        self.stats = {"samples": len(sids), "groups": n_groups,
                      "mod_samples": sum(len(g) for g in self.pools.get(True, [])),
                      "mod_groups": len(self.pools.get(True, [])), "unmod_groups": len(self.pools.get(False, [])),
                      "p_mod": self.p_mod, "weight_exp": weight_exp, "epoch_size": self.epoch_size,
                      "largest_group_share": round(max(
                          (self.p_mod if k else 1 - self.p_mod) * float(p.max()) for k, p in self.probs.items()), 4),
                      "missing_from_csv": sum(s not in info for s in sids)}

    def _next_member(self, k, j):
        # next member of group j of pool k, reshuffling once all members have been used
        if self.order[k][j] is None or self.pos[k][j] >= len(self.order[k][j]):
            self.order[k][j] = self.rng.permutation(self.pools[k][j])
            self.pos[k][j] = 0
        i = self.order[k][j][self.pos[k][j]]
        self.pos[k][j] += 1
        return int(i)

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            k = bool(self.rng.random() < self.p_mod)                    # 1. modified or unmodified pool
            j = self.rng.choice(len(self.pools[k]), p=self.probs[k])    # 2. group, ∝ size**weight_exp
            yield self._next_member(k, j)                               # 3. next member in the group's cycle


def setup_dataloader(config_data, sids_selection_filepath, group_sampling=False):
    # load selected sids
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype('U'))
    # full sample ids as stored in the h5 (1EHZ, 1VY4_AX, 7F5S_L5-CROP3); split files from make_splits.py
    sids_sel = np.array([s.upper() for s in np.atleast_1d(sids_sel)])

    # create dataset
    dataset = Dataset(
        config_data['dataset_filepath'],
        r_noise=config_data['r_noise'],
        virt_cb=config_data['virt_cb'],
        partial=config_data['partial']
    )

    # data selection criteria
    m = select_by_sid(dataset, sids_sel) # select by sids
    m &= select_by_max_ba(dataset, config_data['max_ba'])  # select by max assembly count
    m &= (dataset.sizes[:,0] <= config_data['max_size'])  # select by max size
    m &= (dataset.sizes[:,1] >= config_data['min_num_res'])  # select by min size

    # update dataset selection
    dataset.m &= m

    # define data loader
    if group_sampling:
        sampler = GroupSampler(dataset, config_data['samples_filepath'],
                               p_mod=config_data['p_mod'], epoch_size=config_data['epoch_size'],
                               weight_exp=config_data.get('group_weight_exp', 0.5))
        dataloader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=8, pin_memory=False, prefetch_factor=4)
    else:
        dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=8, pin_memory=False, prefetch_factor=4)

    return dataloader


def dataset_class_ratios(dataset):
    """Share of each label class among all residues of the selected samples, counted once from the h5.

    Classes are the label columns of the model (std_resnames restricted to std_rna, in that order).
    Residues of other types (protein, ions, ...) count in the total: their label row is all zeros,
    as in the batches. Returns (ratios, class names, residue counts).
    """
    cols = [i for i, r in enumerate(dataset.std_resnames) if r in std_rna]
    counts, total = np.zeros(len(cols)), 0
    with h5py.File(dataset.dataset_filepath, 'r') as hf:
        for key in dataset.keys[dataset.m]:
            hgrp = hf['data/structures/' + key]
            n_atoms, n_res = hgrp.attrs['Mr_shape']
            qr, mr = np.array(hgrp['qr']), np.array(hgrp['Mr'])
            # residue of each atom, then residue name of each residue
            atom_res = np.zeros(n_atoms, dtype=np.int64)
            atom_res[mr[:, 0]] = mr[:, 1]
            res_name = np.full(n_res, -1)
            res_name[atom_res[qr[:, 0]]] = qr[:, 1]
            counts += np.array([np.sum(res_name == c) for c in cols])
            total += n_res
    return counts / max(total, 1), dataset.std_resnames[cols], counts


def eval_step(model, device, batch_data, criterion, pos_ratios, pos_weight_factor, global_step):
    # unpack data
    X, q, M, y, rids_sel = [data.to(device) for data in batch_data[:-1]]
    key = batch_data[-1]
    # compute topology
    ids_topk, _, _, _, _ = extract_topology(X, 64)

    # run model
    try:
        z = model.forward(X, ids_topk, q, M)[rids_sel]
    except RuntimeError:
        with open('datasets/error_pdb.txt', 'a', encoding='utf-8') as f:
            f.write(key+'\n')
        print("Error in:",key)
        
    # positive weight from a running estimate of the class shares (as in the original code;
    # updated at every call, training and validation alike)
    pos_ratios += (pt.mean(y,dim=0).detach() - pos_ratios) / (1.0 + np.sqrt(global_step))
    criterion.pos_weight = pos_weight_factor * (1.0 - pos_ratios) / (pos_ratios + 1e-6)
    dloss = criterion(z, y)

    # same weight for every class (G, A, U, C, PSU, 5MC, OMG): baseline run before choosing a class reweighting
    # (dataset_class_ratios gives the training-set class shares for a dataset-level reweighting)
    loss_factors = pt.full((1, dloss.shape[1]), 1.0 / dloss.shape[1], device=dloss.device)
    losses = (loss_factors * dloss) / dloss.shape[0]

    return losses, y.detach(), pt.sigmoid(z).detach()


def scoring(eval_results, device=pt.device('cpu')):
    # compute sum losses and scores for each entry
    sum_losses, scores, accs = [], [], []
    for losses, y, p in eval_results:
        sum_losses.append(pt.sum(losses, dim=0))
        scores.append(bc_scoring(y, p))

        # possible reconstruction accuracy
        m = pt.any(y > 0.5, dim=1)
        if pt.any(m):
            s = pt.argmax(y[m], dim=1)

            # direct reconstruction accuracy
            vmp, imp = pt.max(p, dim=1)

            # accuracies
            accs.append([
                pt.mean(pt.round(p[m,s])),
                pt.mean((imp[m] == s).float()),
            ])

    # average scores
    m_losses = pt.mean(pt.stack(sum_losses, dim=0), dim=0).numpy()
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()
    m_accs = nanmean(pt.tensor(accs)).numpy()

    # pack scores
    scores = {'loss': float(np.sum(m_losses))}
    for i in range(m_losses.shape[0]):
        scores[f'{i}/loss'] = m_losses[i]
        for j in range(m_scores.shape[0]):
            scores[f'{i}/{bc_score_names[j]}'] = m_scores[j,i]

    scores['acc_any'] = m_accs[0]
    scores['acc_max'] = m_accs[1]

    return scores


def logging(logger, writer, scores, global_step, pos_ratios, step_type):
    # debug print
    pr_str = ', '.join([f"{r:.4f}" for r in pos_ratios])
    logger.print(f"{step_type}> [{global_step}] loss={scores['loss']:.4f}, acc_any={scores['acc_any']:.3f}, acc_max={scores['acc_max']:.3f}, pos_ratios=[{pr_str}]")

    # store statistics
    summary_stats = {k:scores[k] for k in scores if not np.isnan(scores[k])}
    summary_stats['global_step'] = int(global_step)
    summary_stats['pos_ratios'] = list(pos_ratios.cpu().numpy())
    summary_stats['step_type'] = step_type
    logger.store(**summary_stats)

    # detailed information
    for key in scores:
        writer.add_scalar(step_type+'/'+key, scores[key], global_step)

    # debug print
    for c in np.unique([key.split('/')[0] for key in scores if len(key.split('/')) == 2]):
        logger.print(f'[{c}] loss={scores[c+"/loss"]:.3f}, ' + ', '.join([f'{sn}={scores[c+"/"+sn]:.3f}' for sn in bc_score_names]))


def train(config_data, config_model, config_runtime, output_path):
    # create logger
    logger = Logger(output_path, 'train')

    # print configuration
    logger.print(">>> Configuration")
    logger.print(config_data)
    logger.print(config_runtime)

    # define device
    device = pt.device(config_runtime['device'])

    # create model
    model = Model(config_model)

    # debug print
    logger.print(">>> Model")
    logger.print(model)
    logger.print(f"> {sum([int(pt.prod(pt.tensor(p.shape))) for p in model.parameters()])} parameters")

    # reload model if configured
    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
    if os.path.isfile(model_filepath) and config_runtime["reload"]:
        logger.print("Reloading model from save file")
        model.load_state_dict(pt.load(model_filepath))
        # get last global step
        global_step = json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['global_step']
        # dynamic positive weight
        pos_ratios = pt.from_numpy(np.array(json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['pos_ratios'])).float().to(device)
    else:
        # starting global step
        global_step = 0
        # dynamic positive weight
        pos_ratios = 0.05*pt.ones(config_model['dm']['N2'], dtype=pt.float).to(device)

    # debug print
    logger.print(">>> Loading data")

    # setup dataloaders
    dataloader_train = setup_dataloader(config_data, config_data['train_selection_filepath'],
                                        group_sampling=config_data.get('group_sampling', False))
    dataloader_validation = setup_dataloader(config_data, config_data['validation_selection_filepath'])

    # debug print
    logger.print(f"> training data size: {len(dataloader_train.dataset)} samples, {len(dataloader_train)} draws per epoch")
    if isinstance(dataloader_train.sampler, GroupSampler):
        logger.print(f"> group sampling: {dataloader_train.sampler.stats}")
    logger.print(f"> validation data size: {len(dataloader_validation)}")

    # debug print
    logger.print(">>> Starting training")

    # send model to device
    model = model.to(device)

    # define losses functions
    criterion = pt.nn.BCEWithLogitsLoss(reduction="none")

    # define optimizer
    optimizer = pt.optim.Adam(model.parameters(), lr=config_runtime["learning_rate"])

    # restart timer
    logger.restart_timer()

    # summary writer
    writer = SummaryWriter(os.path.join(output_path, 'tb'))

    # min loss initial value
    min_loss = 1e9

    # quick training step on largest data: memory check and pre-allocation
    batch_data = dataloader_train.dataset.get_largest()
    optimizer.zero_grad()
    losses, _, _ = eval_step(model, device, batch_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)
    loss = pt.sum(losses)
    loss.backward()
    optimizer.step()
    # start training
    for epoch in range(config_runtime['num_epochs']):
        # train mode
        model = model.train()

        # train model
        train_results = []
        for batch_train_data in tqdm(dataloader_train):
            # print(batch_train_data)
            # skip failed loads and structures with fewer atoms than the max number of neighbors
            if batch_train_data is None or batch_train_data[0].shape[0] < model.max_nn:
                continue

            # global step
            global_step += 1

            # set gradient to zero
            optimizer.zero_grad()

            # forward propagation
            losses, y, p = eval_step(model, device, batch_train_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)
            
            # backward propagation
            loss = pt.sum(losses)
            loss.backward()

            # optimization step
            optimizer.step()

            # store evaluation results
            train_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

            # log step
            if (global_step+1) % config_runtime["log_step"] == 0:
                # process evaluation results
                with pt.no_grad():
                    # scores evaluation results and reset buffer
                    scores = scoring(train_results, device=device)
                    train_results = []

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "train")

                    # save model checkpoint
                    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
                    pt.save(model.state_dict(), model_filepath)

            # evaluation step
            if (global_step+1) % config_runtime["eval_step"] == 0:
                # evaluation mode
                model = model.eval()

                with pt.no_grad():
                    # evaluate model
                    validation_results = []
                    for step_te, batch_validation_data in enumerate(dataloader_validation):
                        # skip failed loads and structures with fewer atoms than the max number of neighbors
                        if batch_validation_data is None or batch_validation_data[0].shape[0] < model.max_nn:
                            continue

                        # forward propagation
                        losses, y, p = eval_step(model, device, batch_validation_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

                        # store evaluation results
                        validation_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                        # stop evaluating
                        if step_te >= config_runtime['eval_size']:
                            break

                    # scores evaluation results
                    scores = scoring(validation_results, device=device)

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "validation")

                    # save model and update min loss
                    if min_loss >= scores['loss']:
                        # update min loss
                        min_loss = scores['loss']
                        # save model
                        model_filepath = os.path.join(output_path, 'model.pt')
                        logger.print("> saving model at {}".format(model_filepath))
                        pt.save(model.state_dict(), model_filepath)

                # back in train mode
                model = model.train()


if __name__ == '__main__':
    # train model
    train(config_data, config_model, config_runtime, '.')
