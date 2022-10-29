from typing import Any

import torch
from tqdm import tqdm

from models.model_wrapper import MLPNetWrapper

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(test_dataloader: Any, config: dict, checkpoint_path: str):
    """
    Predicts new data.

    ----
    Args:
        test_data: Path to csv file containing the paths to the audios files for prediction and its labels.

        batch_size: Mini-Batch size.

        checkpoint_path: Path to the file that contains the saved weight of the model trained.

        num_workers: Number of workers to use as paralel processing.

        use_amp: True to use Mixed precision and False otherwise.
    """
    print(config)
    mlp_net = MLPNetWrapper(config=config).load_from_checkpoint(checkpoint_path)
    mlp_net.to(device)

    mlp_net.freeze()

    pred_list = []
    labels = []

    print(checkpoint_path)

    with torch.no_grad():
        mlp_net.eval()

        for X, y in tqdm(test_dataloader):
            test_audio, test_label = X.to(device), y.to(device)

            out = mlp_net(test_audio)

            pred = torch.argmax(out, axis=1).cpu().detach().numpy()
            label = test_label.cpu().detach().numpy()

            # print(label, pred)

            pred_list.extend(pred)
            labels.extend(label)

    return labels, pred_list