"""
Note that, this specifically trains on TEMPLATE.
"""
from music import Music
import torch
from model import Classifier
from transformers import get_cosine_schedule_with_warmup, AdamW
from trainer import trainer_validator
from report_metric import report_intent, report_slot, report_f1_exact_slot
from music_args import train_args

def setup_optimizer_scheduler(config, param_optimizer):
    num_train_steps = 1000
    num_warmup_steps = 4000
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.learning_rate, eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_train_steps
                                             )
    return optimizer, scheduler

if __name__ == "__main__":
    music_config = train_args().parse_args()
    music_obj = Music(music_config)
    music_hard = Music(read_hard=True)
    test_loader = torch.utils.data.DataLoader(
        music_hard,
        batch_size=music_config.valid_batch_size,
        num_workers=2,
        collate_fn = music_hard.collate_fn,
        shuffle=False
    )
    model = Classifier(music_config)
    device = torch.device("cuda")
    model.to(device)

    bert_optim = list(model.sent_encoder.named_parameters())
    intent_optim = list(model.mlp_intent.named_parameters())
    slot_optim = list(model.mlp_slot.named_parameters())

    optimizers, schedulers = [], []
    for param in [bert_optim, intent_optim, slot_optim]:
        opt, sche = setup_optimizer_scheduler(music_config, param)
        optimizers.append(opt)
        schedulers.append(sche)

    best_info = None
    best_f1 = 0
    for epoch in range(1, music_config.epochs+1):
        if epoch > 1:
            music_obj.generate_train_test_data(is_train=True, loop = 1)
        train_loader = torch.utils.data.DataLoader(
                music_obj,
                batch_size=music_config.train_batch_size,
                num_workers=2,
                collate_fn = music_obj.collate_fn,
                shuffle=True
            )
        train_info, train_f1 = trainer_validator(train_loader, model, optimizers, device, epoch, schedulers)
        
        music_obj.generate_train_test_data(is_train=False, loop = 1)
        valid_loader = torch.utils.data.DataLoader(
                music_obj,
                batch_size=music_config.valid_batch_size,
                num_workers=2,
                collate_fn = music_obj.collate_fn,
                shuffle=True
            )
        valid_info, valid_f1 = trainer_validator(valid_loader, model, optimizers, device, epoch, schedulers, is_train = False)
        print(f"{epoch}. VALID REPORT")
        if music_config.report_intent == True:
            report_intent(valid_info, music_obj)
        if music_config.report_slot == True:
            report_slot(valid_info, music_obj)
        #report_f1_exact_slot(test_info)
        test_info, test_f1 = trainer_validator(test_loader, model, optimizers, device, epoch, schedulers, is_train = False)
        print(f"{epoch}. TEST HARD DATA REPORT")
        if music_config.report_intent == True:
            report_intent(test_info, music_obj)
        if music_config.report_slot == True:
            report_slot(test_info, music_obj)
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_info = test_info
            torch.save(model.state_dict(), "checkpoint_best.pt")