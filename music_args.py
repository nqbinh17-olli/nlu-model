import argparse

def train_args():
    parser = argparse.ArgumentParser(description='Music NLU training')
    # Define Model Config
    parser.add_argument("--train-batch-size", default=64, help="Train batch size", type=int)
    parser.add_argument("--valid-batch-size", default=64, help="Valid batch size", type=int)
    parser.add_argument("--checkpoint-batch-size", default=16, help="Checkpoint batch size", type=int)
    parser.add_argument("--epochs", default=10, help="Training epochs", type=int)
    parser.add_argument("--model-path", default="model.bin", help="Save model name", type=str)
    parser.add_argument("--dropout", default=0.1, help="dropout", type=float)
    parser.add_argument("--bert-dim", default=768, help="Bert dim", type=int)
    parser.add_argument("--learning-rate", default=2e-5, help="Learning rate", type=float)
    parser.add_argument("--train-rate", default=0.7, help="Split train/test rate", type=float)
    parser.add_argument("--is-train", default=True, help="True train, False inference/test", type=bool)
    parser.add_argument("--from-pretrained", default="bert-base-multilingual-cased", help="Define pretrained model", type=str)
    # Define data path
    parser.add_argument("--playmusic-path", default="../input/skillmusic/data/Play Music AI Data.xlsx", help="Load playmusic", type=str)
    parser.add_argument("--musictopic-path", default="../input/skillmusic/data/Music Topic Playlists.xlsx", help="Load musictopic", type=str)
    parser.add_argument("--hard-data-path", default=None, help="Finetuning on hard data", type=str)
    # Define option in Dataset
    parser.add_argument("--is-merge-topic", default=True, help="toggle if want to merge all topics into one", type=bool)
    parser.add_argument("--is-merge-artist-composer", default=True, help="toggle if want to merge artist & composer", type=bool)
    parser.add_argument("--num-slot", default=8, help="Number of slots", type=int)
    parser.add_argument("--num-intent", default=24, help="Number of intents", type=int)
    # Define option for report F1 score & load checkpoint
    parser.add_argument("--report-intent", default=False, help="Report intent", type=bool)
    parser.add_argument("--report-slot", default=False, help="Report slot", type=bool)
    parser.add_argument("--checkpoint", default=None, help="Load checkpoint", type=str)

    
    return parser