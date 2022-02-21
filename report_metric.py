from sklearn.metrics import classification_report
import numpy as np
from collections import namedtuple

def report_intent(info, music_obj):
    y_pred,y_true = [], []
    for d in info:
        y_pred += d["predict_intent"]
        y_true += d["label_intent"]
    num_intent = len(music_obj.intent_label_dict.keys())
    names = list(music_obj.intent_label_dict.keys())
    y_pred = np.array(y_pred).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    print(classification_report(y_true, y_pred, target_names=names))

def report_slot(info, music_obj):
    y_pred,y_true = [], []
    for d in info:
        for arr1, arr2 in zip(d["predict_slot"], d["label_slot"]):
            y_pred += arr1
            y_true += arr2
    num_slot = len(music_obj.slot_label_dict.keys())
    names = list(music_obj.slot_label_dict.keys())
    y_pred = np.array(y_pred).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    print(classification_report(y_true, y_pred, target_names=names))

def calculate_f1_for_each(item):
    print("Exact Slot F1")
    slot_true, decoded_true, slot_pred, decoded_pred = [np.array(i) for i in item]
    keys = set(slot_true)
    store = []
    condition = decoded_true == decoded_pred
    for key in keys:
        if key == "<PAD>":
            continue
        f1 = calculate_f1(slot_true, slot_pred, condition, key)
        store.append(f1)
        print(f"{key}: {f1}")
    print("Macro f1: ", sum(store) / len(store), "\n")
    return
        
def calculate_f1(y_true, y_pred, condition, key):
  select = (y_true == y_pred)[(y_pred == key) & (condition)]
  tp = select.sum()
  fp = len(select) - tp
  fn = (y_true == key)[(y_pred != key)].sum()
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1 = 2.0 / (precision ** -1 + recall ** -1)
  return f1

def report_f1_exact_slot(info, music_obj):
    slot_pred, slot_true, decoded_pred, decoded_true = [[] for _ in range(4)]
    for batch in info:
        num_batch = batch["ids"].size(0)

        for i in range(num_batch):
            text =  batch["text"][i]
            ids = batch["ids"][i]
            label_slot = batch["label_slot"][i]
            predict_slot = batch["predict_slot"][i]

            true = music_obj.decode_exact_slot(ids, label_slot)
            pred = music_obj.decode_exact_slot(ids, predict_slot)
            slot_true += true.decoded_slot
            decoded_true += true.decoded_ids
            slot_pred += pred.decoded_slot
            decoded_pred += pred.decoded_ids
            while len(slot_true) != len(slot_pred):
                if len(slot_true) > len(slot_pred):
                    slot_pred.append("<PAD>")
                    decoded_pred.append("<PAD>")
                else:
                    slot_true.append("<PAD>")
                    decoded_true.append("<PAD>")

    for_cal_f1 = namedtuple("for_cal_f1", ["slot_true", "decoded_true", "slot_pred", "decoded_pred"])
    item = for_cal_f1(slot_true, decoded_true, slot_pred, decoded_pred)
    calculate_f1_for_each(item)