"""
    Code chua xong
"""
import torch
from model import Classifier
from music_args import train_args
from music import Music

def inference(text, model, music_obj, device):
    tokenized = music_obj.tokenizer(text)
    max_len = 0
    ids, type_ids, mask = [], [], []
    for i in range(len(text)):
        max_len = max(max_len, len(tokenized["input_ids"][i]))
        ids.append(tokenized["input_ids"][i])
        type_ids.append(tokenized["token_type_ids"][i])
        mask.append(tokenized["attention_mask"][i])

    def padding(data, max_len):
        pad_len = max_len - len(data)
        data = data + [0] * pad_len
        return data

    def PaddingData(data, max_len):
        padded_data = [padding(d, max_len) for d in data]
        return padded_data
    
    ids = torch.LongTensor(PaddingData(ids, max_len)).to(device)
    type_ids = torch.LongTensor(PaddingData(type_ids, max_len)).to(device)
    mask = torch.LongTensor(PaddingData(mask, max_len)).to(device)
    batch_size = ids.size(0)
    _, score_intent, score_slot = model(ids, mask, type_ids)
    predict_intent = torch.max(torch.softmax(score_intent, dim=-1),dim=1).indices
    predict_slot = torch.max(torch.softmax(score_slot, dim=-1),dim=-1).indices
    predict_slot = predict_slot.reshape(batch_size,-1)

    for i, [txt, idx, slot_label] in enumerate(zip(text, ids, predict_slot)):
        idx = idx.tolist()
        slot_label = slot_label.tolist()
        print(f"{i+1}. Text: {txt} | {music_obj.decode_exact_slot(idx, slot_label)} | Pred: {slot_label}")
        print("\n")
    return

if __name__ == "__main__":
    music_config = train_args().parse_args()
    music_obj = Music(music_config)
    model = Classifier(music_config)
    device = torch.device("cuda")
    model.to(device)

    inference(iter_data, model, music_obj, device)