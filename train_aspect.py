# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset,MapDataset
from paddlenlp.metrics import AccuracyAndF1

from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
from sklearn.model_selection import KFold, RepeatedKFold

from attack_train import *
from utils import evaluate2
from logger import Logger
log = Logger().get_logger()

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=1e-6, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=15, type=int, help="Total number of training epochs to perform.")#50
parser.add_argument("--init_from_ckpt", type=bool, default=False, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=2021, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--train_data", type=str, default="SE-ABSA16_PHNS", help="The train datasets to be loaded.")
parser.add_argument("--fold_index", type=int, default=0, help="select CV number")
parser.add_argument("--nfold", type=int, default=3, help="set Fold number")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """Sets random seed."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False,
                    dataset_name="chnsenticorp"):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence has the following format:
    ::
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence pair mask has the following format:
    ::

        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
    
    note: There is no need token type ids for skep_roberta_large_ch model.


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.
        dataset_name((obj:`str`, defaults to "chnsenticorp"): The dataset name, "chnsenticorp" or "sst-2".

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    def truncate(s, ratio=0.6,max_len=512-1-3-len(example["text"])):
        head_len = int(max_len*ratio)
        tail_len = max_len - head_len
        return s[:head_len] +"#" + s[-tail_len:]
    
    encoded_inputs = tokenizer(
        text=example["text"],
        text_pair=truncate(example["text_pair"]),
        max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

def get_preprocess_data(train_ds):
    tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(dtype="int64")  # labels
    ): [data for data in fn(samples)]

    data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    return data_loader

def get_preprocess_model(train_ds):
    model = SkepForSequenceClassification.from_pretrained(
        'skep_ernie_1.0_large_ch', num_classes=len(train_ds.label_list))

    if args.init_from_ckpt:# and os.path.isfile(args.init_from_ckpt):
        # '''==========================用CAME的模型精调PHNS=========================='''
        train_data = "SE-ABSA16_CAME1"
        ckpt_path = "checkpoint/"+ args.train_data +"/"+ str(args.fold_index)+"/best.pdparam"
        state_dict = paddle.load(ckpt_path)
        model.set_dict(state_dict)
        log("Load checkpoint from {}".format(ckpt_path))
    model = paddle.DataParallel(model)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
    return model, optimizer

def run_cv(train_ds,rank, fold_index, nfold=3, data_test=None,random_state=args.seed):
    folds = KFold(n_splits=nfold, shuffle=True,random_state=random_state)
    tnp_ds = np.array(train_ds.data)
    log("Dataset: {}, Total Folds is {}".format(args.train_data, nfold))
    for fold, (tidx, vidx) in enumerate(folds.split(tnp_ds)):
        if fold !=fold_index:
            continue
        log("=====================Fold Num {}=====================".format(fold))
        tds = MapDataset(tnp_ds[tidx].tolist())
        vds = MapDataset(tnp_ds[vidx].tolist())

        train_data_loader = get_preprocess_data(tds)
        test_data_loader = get_preprocess_data(vds)

        model,optimizer = get_preprocess_model(train_ds)
        #metric = ChunkEvaluator(label_list=train_ds.label_list, suffix=True)

        #num_training_steps = len(train_data_loader) * args.epochs
        criterion = paddle.nn.loss.CrossEntropyLoss()
        metric = paddle.metric.Accuracy()
        
        test_metric = AccuracyAndF1()

        global_step = 0
        tic_train = time.time()
        best_score = 0

        # attack_train_mode = "fgm"
        # if attack_train_mode == 'fgm':
        #     att_fun = FGM(model=model)
        # elif attack_train_mode == 'pgd':
        #     att_fun = PGD(model=model)

        ########################Start Run Model ##############################
        for epoch in range(1, args.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                input_ids, token_type_ids, labels = batch
                logits = model(input_ids, token_type_ids)
                loss = criterion(logits, labels)
                probs = F.softmax(logits, axis=1)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                global_step += 1
                if global_step % 10 == 0 and rank == 0:
                    log(
                        "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, step, loss, acc,
                        10 / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()

                attack(att_fun, model, optimizer, batch, use_n_gpus=True)

                optimizer.step()
                optimizer.clear_grad()
            if rank == 0:
                eval_loss ,res = evaluate2(model,criterion, test_metric,test_data_loader)
                log(
                    "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
                    % (
                        eval_loss.numpy(),
                        res[0],
                        res[1],
                        res[2],
                        res[3],
                        res[4]))
                if res[4] > best_score:
                    best_score = res[4]
                    # save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                    save_dir = args.save_dir+"/"+args.train_data +"/"+str(fold)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    file_name = os.path.join(save_dir, "best.pdparam")
                    # Need better way to get inner model of DataParallel
                    paddle.save(model._layers.state_dict(), file_name)
                    log("At Fold {}, New best f1 score ===: {}".format(fold,best_score))
        if rank ==0:
            log("At Final Fold {},best f1 score ===: {}".format(fold,best_score))


if __name__ == "__main__":
    log("args setting:\n{}\n".format(args))
    set_seed(args.seed)
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    #train_ds0 = load_dataset("seabsa16", "phns", splits=["train"])
    #args.train_data = "SE-ABSA16_PHNS"#short_
    #args.train_data = "SE-ABSA16_CAME"#short_
    data_path = "data/"+ args.train_data +"/train.tsv"
    train_ds = load_dataset("seabsa16","phns", data_files=data_path)
    
    run_cv(train_ds,rank,args.fold_index,nfold = args.nfold)

