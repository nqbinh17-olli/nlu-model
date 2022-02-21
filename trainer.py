import gc
import traceback
from metric import MetricsMeter, F1Score
import torch
from tqdm.autonotebook import tqdm

def to_one_hot_vector(label):
    batch = label.shape[0]
    if len(label.shape) == 2:
        one_hot = label == 0
    else:
        one_hot = torch.zeros(batch, max(label) + 1)
        one_hot[torch.arange(one_hot.shape[0]), label] = 1
    return one_hot

def trainer_process(data, model, device, metrics_intent, metrics_slot, f1_intent, f1_slot):
    text = data['text']
    template = data['template']
    ids = data['ids']
    mask = data['mask']
    label_intent = data['intent']
    label_slot = data['slot_label'].reshape(-1)
    types = data["types"]
    
    mask = mask.to(device)
    ids = ids.to(device)
    label_intent = label_intent.to(device)
    label_slot = label_slot.to(device)
    types = types.to(device)

    loss, score_intent, score_slot = model(ids, mask, types, label_intent, label_slot)
    with torch.no_grad():
        # predict & compute f1-score
        predict_intent = torch.max(torch.softmax(score_intent, dim=-1),dim=1).indices
        predict_slot = torch.max(torch.softmax(score_slot, dim=-1),dim=-1).indices
        macro_f1_intent, _ = f1_intent(predict_intent, label_intent, 'macro')
        macro_f1_slot, _ = f1_slot(predict_slot, label_slot, 'macro')
        

        # update loss & f1-score
        metrics_intent.update(loss=loss.data.item(), macro_f1=macro_f1_intent.data.item())
        metrics_slot.update(loss=loss.data.item(), macro_f1=macro_f1_slot.data.item())
    
    batch = ids.size(0)
    def convert_to_form(tensor):
        return tensor.detach().cpu().reshape(batch, -1).tolist()
    
    info = {"ids": ids.detach().cpu(), "text": text,
            "label_intent": convert_to_form(label_intent),
            "label_slot": convert_to_form(label_slot),
            "predict_intent": convert_to_form(predict_intent),
            "predict_slot": convert_to_form(predict_slot)
           }
    return loss, info

def trainer_validator(data_loader, model, optimizers, device, epoch, schedulers=None, is_train=True):
    if is_train == True:
        model.train()
    else:
        model.eval()
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    metrics_intent, metrics_slot = MetricsMeter(), MetricsMeter()
    f1_intent, f1_slot = F1Score(), F1Score()
    info_data = []
    with torch.no_grad() if is_train == False else torch.enable_grad():
        for data in tk0:
            gc.collect()
            torch.cuda.empty_cache()
            model.zero_grad()
            loss, info = trainer_process(data, model, device, metrics_intent, metrics_slot, f1_intent, f1_slot)
            info_data.append(info)
            if is_train == True:
                loss.backward()
                for optimizer, scheduler in zip(optimizers, schedulers):
                    optimizer.step()
                    scheduler.step()

            tk0.set_postfix(Epoch=epoch,
                            Train_Loss = metrics_intent.avg_loss,
                            F1_Intent = metrics_intent.avg_macro_f1,
                            F1_Slot = metrics_slot.avg_macro_f1,
                           )
    info = "Train" if is_train else "Test"
    return info_data, metrics_intent.avg_macro_f1