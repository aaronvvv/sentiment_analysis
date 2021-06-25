import pandas as pd
import os
from collections import defaultdict
import argparse
import csv
from logger import Logger
logger = Logger()
#log = logger.get_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--dsname", type=str, default="COTE_BD", help="The path to model parameters to be loaded.")
parser.add_argument("--type", type=str, default="opinion", help="The task type.")
parser.add_argument("--nfold", default=5, type=int, help="Batch size per GPU/CPU for training.")
args = parser.parse_args()

def merge_res(dps,dsname=None):
    res = dps[0].copy(deep=True)
    #nflod = len(dps)
    dbest_cnt = defaultdict(int)
    max_len_fold_dict = defaultdict(int)
    special_cnt = 0
    for r in range(res.shape[0]):
        rows_pred = [str(d.iloc[r,1]).strip() for d in dps]
        #vote = defaultdict(int ,zip(rows_pred,[0]*len(rows_pred)))
        vote = dict(zip(rows_pred,[0]*len(rows_pred)))
        best_cnt = 0
        best = "" 
        max_len = 0
        max_len_pred = ""
        max_len_fold = None

        for i, p in enumerate(rows_pred):
            if dsname[:4]=="COTE" and (p == "nan" or len(p)<=1):continue
            vote[p] += 1
            if vote[p]>best_cnt:
                best_cnt = vote[p]
                best = p
            if dsname[:4]=="COTE" and len(p)>max_len:
                max_len = len(p)
                max_len_pred = p
                max_len_fold = i
        if best_cnt ==1:
            best = max_len_pred
            log("best_cnt ==1 at Line {},best: {}".format(r,best))
        if dsname[:4]=="COTE" and best != max_len_pred:
            max_len_fold_dict[max_len_fold] += 1
            log("best != max_len_pred --->Line:{}, best: {},cnt:{} ;max_len_pred: {},cnt:{},fold:{}".format(r,best,best_cnt,max_len_pred,vote[max_len_pred],max_len_fold))
            if best_cnt<=vote[max_len_pred]:
                best = max_len_pred
            # elif (args.dsname[-2:] == "DP" and max_len_pred[-1:] in ["店","家","厅","面","菜","馆","茶"]) \
            #     or (args.dsname[-2:] == "FW"  and  max_len_pred[-1:] in ["园","区","寺","庙","台"]):
            #     best = max_len_pred
            #     special_cnt += 1
        if best_cnt == 0:
            log("Line {} no predict!!!!!!!!!!!!!!!!!!!!!".format(r)) 
            
        dbest_cnt[best_cnt] += 1 
        res.iloc[r,1] = best
    log(f"special_cnt: {special_cnt} \n max_len_fold_dict : \n{max_len_fold_dict}\n")
    if dsname:
        res_dir = "./results"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        res_path = "results/" + dsname + ".tsv"
        res.to_csv(res_path,sep="\t",index=False)
    log("Dataset {} , best count: {}".format(dsname, dbest_cnt))
    return res

def get_merge_res(dsname,nflod=3):
    dps = []
    for i in range(nflod):
        res_path = "checkpoint/" + dsname  + "/" + str(i)+ "/" + dsname + ".tsv"
        dps.append(pd.read_csv(res_path,sep="\t",quoting=csv.QUOTE_NONE))
    merge_res(dps,dsname)

#python vote.py --dsname COTE_MFW --nfold 6
#python vote.py --dsname COTE_BD --nfold 5
#python vote.py --dsname COTE_DP --nfold 5

log = None
if __name__ == "__main__":
    log = logger.new_file_logger(args.dsname)
    log("\n============================Vote for: {}, Fold num:{}============================\n".format(args.dsname,args.nfold))
    get_merge_res(args.dsname, args.nfold)
