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
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
from logger import Logger
from tqdm import tqdm

log = Logger().get_logger()
# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default=None, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--model_name', choices=['skep_ernie_1.0_large_ch', 'skep_ernie_2.0_large_en'],
    default="skep_ernie_1.0_large_ch", help="Select which model to train, defaults to skep_ernie_1.0_large_ch.")
parser.add_argument("--predict_data", type=str, default="NLPCC14-SC", help="The test datasets to be predicted.")
parser.add_argument("--param_num", default=0, type=int, help="Params checkpoint number.")
args = parser.parse_args()
# yapf: enable


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=True,
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
    if dataset_name == "sst-2":
        encoded_inputs = tokenizer(
            text=example["sentence"], max_seq_len=max_seq_length)
    elif dataset_name in[ "ChnSentiCorp","NLPCC14-SC"]:
        encoded_inputs = tokenizer(
            text=example["text"], max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        if dataset_name == "sst-2":
            label = np.array([example["labels"]], dtype="int64")
        elif dataset_name in[ "ChnSentiCorp","NLPCC14-SC"]:
            label = np.array([example["label"]], dtype="int64")
        else:
            raise RuntimeError(
                f"Got unkown datatset name {dataset_name}, it must be processed on your own."
            )

        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

@paddle.no_grad()
def predict1(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        input_ids, token_type_ids = convert_example(
            text,
            tokenizer,
            label_list=label_map.values(),
            max_seq_length=args.max_seq_length,
            is_test=True)
        examples.append((input_ids, token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token type ids
    ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results

@paddle.no_grad()
def predict(model, test_data_loader):
    label_map = {0: '0', 1: '1'}
    results = []
    # 切换model模型为评估模式，关闭dropout等随机因素
    model.eval()
    for batch in tqdm(test_data_loader):
        input_ids, token_type_ids = batch
        # 喂数据给模型
        logits = model(input_ids, token_type_ids)
        # 预测分类
        probs = F.softmax(logits, axis=-1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
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
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

def get_preprocess_data(test_ds,mode="test"):
    tokenizer = SkepTokenizer.from_pretrained(args.model_name)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_name=args.predict_data)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    ): [data for data in fn(samples)]

    data_loader = create_dataloader(
        test_ds,
        mode=mode,
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    return data_loader

if __name__ == "__main__":
    paddle.set_device(args.device)

    # These data samples is in Chinese.
    # If you use the english model, you should change the test data in English.
    # data = [
    #     '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般',
    #     '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
    #     '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    # ]
    #label_map = {0: 'negative', 1: 'positive'}
    #test_ds = load_dataset("nlpcc14_sc",data_files=data_path)
 
    if args.predict_data == "ChnSentiCorp":
        test_ds = load_dataset("chnsenticorp",splits=["test"])
    elif args.predict_data == "NLPCC14-SC": 
        test_ds = load_dataset("nlpcc14_sc",splits=["test"])
    test_data_loader = get_preprocess_data(test_ds)

    label_map = {0: '0', 1: '1'}
    model = SkepForSequenceClassification.from_pretrained(
        args.model_name, num_classes=len(label_map))
    tokenizer = SkepTokenizer.from_pretrained(args.model_name)

    res_dir = None
    if args.params_path is None: 
        res_dir = "./checkpoint/"+args.predict_data +"/" +str(args.param_num) 
        state_dict = paddle.load(res_dir+"/best.pdparam")
        model.set_dict(state_dict)
        log("Loaded parameters from %s" % res_dir)
    else:
        res_dir = args.params_path
        state_dict = paddle.load(res_dir+"/best.pdparam")
        model.set_dict(state_dict)
        log("Loaded parameters from %s" % args.params_path)
    #results = predict( model, data, tokenizer, label_map, batch_size=args.batch_size)
    results = predict( model,test_data_loader)
    
    log("{} Predict End!".format(args.predict_data))
    with open(os.path.join(res_dir, args.predict_data+".tsv"), 'w', encoding="utf8") as f:
        f.write("index\tprediction\n")
        for idx, example in enumerate(results):
            f.write(str(idx)+"\t"+results[idx]+"\n")
