import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
@paddle.no_grad()
def evaluate(model, metric, test_data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in test_data_loader:
        preds = model(input_ids, seg_ids,seq_lens = lens)
        #preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()         
    model.train()
    return precision, recall, f1_score

@paddle.no_grad()
def evaluate2(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    # if isinstance(metric, AccuracyAndF1):
    #     print(
    #         "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
    #         % (
    #             loss.numpy(),
    #             res[0],
    #             res[1],
    #             res[2],
    #             res[3],
    #             res[4], ),
    #         end='')
    # elif isinstance(metric, Mcc):
    #     print("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]), end='')
    # elif isinstance(metric, PearsonAndSpearman):
    #     print(
    #         "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
    #         % (loss.numpy(), res[0], res[1], res[2]),
    #         end='')
    # else:
    #     print("eval loss: %f, acc: %s, " % (loss.numpy(), res), end='')
    model.train()
    return loss, res

from paddlenlp.datasets import MapDataset
def load_datasets(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            wrong_list = []
            for line in fp.readlines():
                try:
                    names, words = line.strip('\n').split('\t')
                except Exception as e:
                    wrong_list.append(line)
                    continue
                labels = []
                len_names = len(names)
                len_words = len(words)
                for i in range(len_words-len_names+1):
                    if words[i]==names[0] and words[i:i+len_names] ==names:
                        labels.extend([0]+[1]*(len_names-1))
                        labels.extend([2]*(len_words-i-len_names))
                        break
                    else:
                        labels.append(2)
                words = list(words)
                if wrong_list:
                    print(wrong_list)
                yield {'tokens': words,"labels":labels}#, 'idx': idx

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]
