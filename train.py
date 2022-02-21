from music import Music
import torch
from model import Classifier
from transformers import get_cosine_schedule_with_warmup, AdamW
from trainer import trainer_validator
from report_metric import report_intent, report_slot, report_f1_exact_slot
import argparse

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

class music_config:
    playmusic_path = "../input/skillmusic/data/Play Music AI Data.xlsx"
    musictopic_path = "../input/skillmusic/data/Music Topic Playlists.xlsx"
    is_merge_topic = True
    is_merge_artist_composer = True
    cut_by_intents = ["song", "artist", "topic", "sort_by", "loop_song_duration", "loop_song_timepoint",
                     "general", "song_artist", "song_composer", "composer", "loop_song", "loop_song_times",
                     "loop_current", "loop_current_duration", "loop_current_timepoint", "loop_current_times",
                     "artist_duration", "artist_timepoint", "topic_duration", "topic_timepoint", "general_duration",
                     "general_timepoint", "num_topic", "num_artist"]

class model_config:
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 64
    checkpoint_batch_size = 16
    EPOCHS = 10
    MODEL_PATH = "model.bin"
    dropout = 0.1
    bert_dim = 768
    learning_rate = 2e-5
    train_rate = 0.7

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Music NLU training')
    parser.add_argument("--train-batch-size", default=64, help="Train batch size", type=int)
    parser.add_argument("--valid-batch-size", default=64, help="Valid batch size", type=int)
    parser.add_argument("--checkpoint-batch-size", default=16, help="Checkpoint batch size", type=int)
    parser.add_argument("--epochs", default=10, help="Training epochs", type=int)
    parser.add_argument("--model-path", default="model.bin", help="Save model name", type=str)
    parser.add_argument("--dropout", default=0.1, help="dropout", type=float)
    parser.add_argument("--bert-dim", default=768, help="Bert dim", type=int)
    parser.add_argument("--learning-rate", default=2e-5, help="Learning rate", type=float)
    parser.add_argument("--train-rate", default=0.7, help="Split train/test rate", type=float)

    parser.add_argument("--playmusic-path", default="../input/skillmusic/data/Play Music AI Data.xlsx", help="Save model name", type=str)
    parser.add_argument("--musictopic-path", default="../input/skillmusic/data/Music Topic Playlists.xlsx", help="Save model name", type=str)
    parser.add_argument("--is-merge-topic", default=True, help="toggle if want to merge all topics into one", type=bool)
    parser.add_argument("--is-merge-artist-composer", default=True, help="toggle if want to merge artist & composer", type=bool)

    args = parser.parse_args()

    music_obj = Music(music_config)
    music_obj.cut_to_small_data()
    music_obj.prepare_slot_dict()
    music_obj.prepare_train_test()
    music_obj.generate_train_test_data(is_train=True, loop=1)

    num_slot = len(music_obj.slot_label_dict.keys())
    num_intent = len(music_obj.intent_label_dict.keys())
    model = Classifier(num_intent, num_slot)
    device = torch.device("cuda")
    model.to(device)

    bert_optim = list(model.sent_encoder.named_parameters())
    intent_optim = list(model.mlp_intent.named_parameters())
    slot_optim = list(model.mlp_slot.named_parameters())

    optimizers, schedulers = [], []
    for param in [bert_optim, intent_optim, slot_optim]:
        opt, sche = setup_optimizer_scheduler(model_config, param)
        optimizers.append(opt)
        schedulers.append(sche)

    best_info = None
    best_f1 = 0
    for epoch in range(1, model_config.EPOCHS+1):
        if epoch > 1:
            music_obj.generate_train_test_data(is_train=True, loop = 1)
        train_loader = torch.utils.data.DataLoader(
                music_obj,
                batch_size=32,
                num_workers=2,
                collate_fn = music_obj.collate_fn,
                shuffle=True
            )
        train_info, _ = trainer_validator(train_loader, model, optimizers, device, epoch, schedulers)
        
        music_obj.generate_train_test_data(is_train=False, loop = 1)
        valid_loader = torch.utils.data.DataLoader(
                music_obj,
                batch_size=model_config.VALID_BATCH_SIZE,
                num_workers=2,
                collate_fn = music_obj.collate_fn,
                shuffle=True
            )
        test_info, f1 = trainer_validator(valid_loader, model, optimizers, device, epoch, schedulers, is_train = False)
        report_intent(test_info, music_obj)
        report_slot(test_info, music_obj)
        #report_f1_exact_slot(test_info)
        if f1 > best_f1:
            best_f1 = f1
            best_info = test_info
            torch.save(model.state_dict(), "checkpoint_best.pt")