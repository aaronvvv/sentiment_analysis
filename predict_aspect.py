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
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
from tqdm import tqdm
from logger import Logger

log = Logger().get_logger()

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default=None, help="The path to model parameters to be loaded.")
parser.add_argument("--param_name", type=str, default=None, help="The name of model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for prediction.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--predict_data", type=str, default="SE-ABSA16_PHNS", help="The test datasets to be predicted.")
parser.add_argument("--param_num", default=0, type=int, help="Params checkpoint number.")

args = parser.parse_args()
# yapf: enable


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
    for batch in tqdm(data_loader):
        input_ids, token_type_ids = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


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

from paddlenlp.datasets import MapDataset
def load_datasets(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                idx, text_a, text_b = line.strip('\n').split('\t')
                yield {'text': text_a,'text_pair':text_b}

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


if __name__ == "__main__":
    paddle.set_device(args.device)
    #test_ds = load_dataset("seabsa16", "phns", splits=["test"])
    data_path = "data/"+ args.predict_data +"/test.tsv"
    test_ds = load_datasets(datafiles=data_path)
    #test_ds1 = load_dataset("seabsa16","phns", data_files=data_path,splits=["test"])

    #label_map = {0: 'negative', 1: 'positive'}
    label_map = {0: '0', 1: '1'}

    model = SkepForSequenceClassification.from_pretrained(
        'skep_ernie_1.0_large_ch', num_classes=len(label_map))
    tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    ): [data for data in fn(samples)]

    test_data_loader = create_dataloader(
        test_ds,
        mode='test',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)
   
    res_dir = None
    if args.params_path is None:
        res_dir = "./checkpoint/"+args.predict_data +"/" +str(args.param_num)
        state_dict = paddle.load(res_dir+"/best.pdparam")
        model.set_dict(state_dict)
        log("Loaded parameters from %s" % res_dir)
    else:
        res_dir = args.params_path
        state_dict = paddle.load(res_dir+ "/" +args.param_name)
        model.set_dict(state_dict)
        log("Loaded parameters from %s" % res_dir)

    results = predict(model, test_data_loader, label_map)
    log("{} Predict End!".format(args.predict_data))
    with open(os.path.join(res_dir, args.predict_data+".tsv"), 'w', encoding="utf8") as f:
        f.write("index\tprediction\n")
        for idx, example in enumerate(results):
            f.write(str(idx)+"\t"+results[idx]+"\n")

    # for idx, text in enumerate(test_ds.data):
    #     log('Label: {} \t Data: {}'.format(results[idx], text))
