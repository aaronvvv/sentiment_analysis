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
import gc

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset,MapDataset
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import SkepCrfForTokenClassification, SkepModel, SkepTokenizer
from utils import evaluate #convert_example,  predict

from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from logger import Logger
log = Logger().get_logger()

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-7, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=22, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--init_from_ckpt", type=bool, default=True, help="Use checkpoint to initialize model, or not.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--train_data", type=str, default="COTE_DP", help="The train datasets to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument("--fold_index", type=int, default=0, help="select CV number")
parser.add_argument("--nfold", type=int, default=5, help="set Fold number")

args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """Sets random seed."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def convert_example_to_feature(example,
                               tokenizer,
                               max_seq_len=512,
                               no_entity_label="O",
                               is_test=False):
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

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        no_entity_label(obj:`str`, defaults to "O"): The label represents that the token isn't an entity. 
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`list[int]`, optional): The input label if not test data.
    """
    tokens = example['tokens']
    labels = example['labels']
    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = tokenized_input['seq_len']

    if is_test:
        return input_ids, token_type_ids, seq_len
    else:
        labels = labels[:(max_seq_len - 2)]
        encoded_label = np.array(
            [no_entity_label] + labels + [no_entity_label], dtype="int64")

        return input_ids, token_type_ids, seq_len, encoded_label


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

@paddle.no_grad()
def predict(model, data_loader, label_map):
    """
    Given a prediction dataset, it gives the prediction results.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
    """
    model.eval()
    results = []
    for input_ids, token_type_ids, seq_lens in data_loader:
        preds = model(input_ids, token_type_ids, seq_lens=seq_lens)
        tags = parse_predict_result(preds.numpy(), seq_lens.numpy(), label_map)
        results.extend(tags)
    return results


def parse_predict_result(predictions, seq_lens, label_map):
    """
    Parses the prediction results to the label tag.
    """
    pred_tag = []
    for idx, pred in enumerate(predictions):
        seq_len = seq_lens[idx]
        # drop the "[CLS]" and "[SEP]" token
        tag = [label_map[i] for i in pred[1:seq_len - 1]]
        pred_tag.append(tag)
    return pred_tag

def get_preprocess_data(train_ds):
    label_map = {0: "B", 1: "I", 2: "O"}
    # `no_entity_label` represents that the token isn't an entity. 
    no_entity_label_idx = label_map.get("O", 2)
    tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

    trans_func = partial(
        convert_example_to_feature,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        no_entity_label=no_entity_label_idx,
        is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # token type ids
        Stack(dtype='int64'),  # sequence lens
        Pad(axis=0, pad_val=no_entity_label_idx)  # labels
    ): [data for data in fn(samples)]

    data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    return data_loader

def get_preprocess_model(train_ds):
    skep = SkepModel.from_pretrained('skep_ernie_1.0_large_ch')
    model = SkepCrfForTokenClassification(skep, num_classes=len(train_ds.label_list))
    if args.init_from_ckpt:# and os.path.isfile(args.init_from_ckpt):
        ckpt_path = "checkpoint/"+ args.train_data +"/final.pdparam"
        #ckpt_path = "checkpoint/"+ args.train_data +"/model_25260/model_state.pdparam"
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


def run(train_ds,nfold=5,random_state=2021,fold=0):
    train_data_loader = get_preprocess_data(train_ds)
    folds = KFold(n_splits=nfold, shuffle=True,random_state=random_state)
    tnp_ds = np.array(train_ds.data)
    res = list(folds.split(tnp_ds))
    tidx, vidx = res[0]
    vds = MapDataset(tnp_ds[vidx].tolist())
    test_data_loader = get_preprocess_data(vds)

    model,optimizer = get_preprocess_model(train_ds)
    metric = ChunkEvaluator(label_list=train_ds.label_list, suffix=True)
    global_step = 0
    tic_train = time.time()
 
    ########################Start Run Model ##############################
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, seq_lens, labels = batch
            loss = model(input_ids, token_type_ids, seq_lens=seq_lens, labels=labels)
            avg_loss = paddle.mean(loss)
            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                log(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss,
                    10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if rank == 0 and global_step>15000 and global_step % 100 == 0 :
                p,r,f1 = evaluate(model,metric,test_data_loader)
                log("eval precision: %f - recall: %f - f1: %f" % (p, r, f1))
                if f1 > best_score:
                    best_score = f1
                    # save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                    save_dir = args.save_dir+"/"+args.train_data
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    file_name = os.path.join(save_dir, "all_best.pdparam")
                    # Need better way to get inner model of DataParallel
                    paddle.save(model._layers.state_dict(), file_name)
                    log("At Fold {}, New best score ===: {}".format(fold,best_score)) 

    save_dir = args.save_dir+"/"+args.train_data 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = args.save_dir+"/"+args.train_data 
    paddle.save(model._layers.state_dict(), os.path.join(save_dir, "final_last.pdparam"))
    log("At global_step {} save model".format(global_step))

#from utils import load_datasets
from utils import evaluate
if __name__ == "__main__":
    log("args setting:\n{}\n".format(args))
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    
    data_path = "data/"+ args.train_data +"/train.tsv"
    train_ds = load_dataset("cote","dp", data_files=data_path)
    run(train_ds,args.nfold)