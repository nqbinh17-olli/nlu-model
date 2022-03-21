from music import Music
import torch
from model import Classifier
from trainer import trainer_validator
from report_metric import report_intent, report_slot, report_f1_exact_slot
from music_args import train_args

if __name__ == "__main__":
    music_config = train_args().parse_args()
    music_obj = Music(music_config)
    model = Classifier(music_config)
    device = torch.device("cuda")
    model.to(device)
    if music_config.checkpoint is not None:
        try:
            model.load_state_dict(torch.load(music_config.checkpoint))
            model.eval()
            print("Successfully load model checkpoint: ", music_config.checkpoint)
        except:
            print("Fail to load: ", music_config.checkpoint)
    
    valid_loader = torch.utils.data.DataLoader(
                music_obj,
                batch_size=music_config.valid_batch_size,
                num_workers=2,
                collate_fn = music_obj.collate_fn,
                shuffle=True
            )
    test_info, f1 = trainer_validator(valid_loader, model, None, device, 0, None, is_train = False)
    if music_config.report_intent == True:
        report_intent(test_info, music_obj)
    if music_config.report_slot == True:
        report_slot(test_info, music_obj)