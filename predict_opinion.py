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

import argparse
import os
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import SkepCrfForTokenClassification, SkepModel, SkepTokenizer
from tqdm import tqdm

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default="checkpoint/COTE_BD/0/best.pdparam", help="The path to model parameters to be loaded.")
parser.add_argument("--predict_data", type=str, default="COTE_BD", help="The test datasets to be predicted.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=240, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--param_num", default=0, type=int, help="Params checkpoint number.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
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

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask. 
    """
    tokens = example["tokens"]
    #tokens = example[0]
    encoded_inputs = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    seq_len = encoded_inputs["seq_len"]
    return input_ids, token_type_ids, seq_len


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
    results2 = []
    lens = []
    for input_ids, token_type_ids, seq_lens in tqdm(data_loader):
        preds = model(input_ids, token_type_ids, seq_lens=seq_lens)
        tags = parse_predict_result(preds.numpy(), seq_lens.numpy(), label_map)
        lens.extend(seq_lens.numpy().tolist())
        results.extend(tags)
    results = parse_decodes(data_loader,results,lens)
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

def parse_decodes(ds, decodes, lens):
    results = []
    for idx, end in enumerate(lens):
        sent = ds.dataset.data[idx]['tokens'][:end]
        tags = decodes[idx][:end]
        words = ""
        flag = False
        for s, t in zip(sent, tags):
            if t=="B":
                words = s
            elif t == 'I':
                words += s
            else:
                if len(words)>0:
                    flag = True
                    results.append(words)
                    break
        if words=="":
            results.append("")
        elif not flag:
            results.append(words)
    return results

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
        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=batchify_fn,
            return_list=True)
    else:
        #batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        return paddle.io.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            return_list=True,
            collate_fn=batchify_fn)

        
from paddlenlp.datasets import MapDataset
def load_datasets(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                idx, words = line.strip('\n').split('\t')
                idx = idx.split('\002')
                words = words.split('\002')
                words = list(words[0])
                #yield words, idx
                yield {'tokens': words}#, 'idx': idx

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

def get_predict():
    pass
    #test_ds0 = load_dataset("cote", "dp", splits=['test'])
    #test_ds = load_dataset("cote","dp", data_files="data/"+ args.predict_data +"/test.tsv")
    #test_ds = load_dataset("cote","dp", data_files="data/COTE-MFW/test.tsv")
    #test_ds2 = load_dataset(read, data_path=data_path,lazy=False)
from logger import Logger
log = Logger().get_logger()
if __name__ == "__main__":
    paddle.set_device(args.device)

    # Create dataset, tokenizer and dataloader.
    data_path = "data/"+ args.predict_data +"/test.tsv"
    test_ds = load_datasets(datafiles=data_path)

    # The COTE_DP dataset labels with "BIO" schema.
    label_map = {0: "B", 1: "I", 2: "O"}
    # `no_entity_label` represents that the token isn't an entity. 
    no_entity_label_idx = 2

    skep = SkepModel.from_pretrained('skep_ernie_1.0_large_ch')
    model = SkepCrfForTokenClassification(
        skep, num_classes=len(label_map))
    tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

    res_dir = "./checkpoint/"+args.predict_data +"/" +str(args.param_num)
    params_path = res_dir
    if params_path :
        state_dict = paddle.load(res_dir+"/best.pdparam")
        model.set_dict(state_dict)
        log("Loaded parameters from %s" % params_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)
    #test_ds.map(trans_func)
    
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # token type ids
        Stack(dtype='int64'),  # sequence lens
    ): [data for data in fn(samples)]

    test_data_loader = create_dataloader(
        test_ds,
        mode='test',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    results = predict(model, test_data_loader, label_map)
    # 写入预测结果
    # res_dir = "./results"
    # if not os.path.exists(res_dir):
    #     os.makedirs(res_dir)

    with open(os.path.join(res_dir, args.predict_data+".tsv"), 'w', encoding="utf8") as f:
        f.write("index\tprediction\n")
        for idx, example in enumerate(results):
            f.write(str(idx)+"\t"+results[idx]+"\n")
