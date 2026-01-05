from datetime import datetime

import util
import argparse
import torch
from model import HSTT_TASM
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="hidden channels")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument( "--checkpoint", type=str, help="path to best_model.pth")
parser.add_argument("--plotheatmap", type=str, default="True", help="")
args = parser.parse_args()


def main():


    device = torch.device(args.device)

    # >>> 论文一致命名：HSTT-TASM
    model = HSTT_TASM(
        device=device,
        input_dim=args.input_dim,
        channels=args.channels,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        dropout=args.dropout,
    )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print("model load successfully")

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]

    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for it, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae, amape, awmape, armse = [], [], [], []

    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = (
            "Evaluate best model on test data for horizon {:d}, "
            "Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
        )
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(args.output_len, np.mean(amae), np.mean(amape), np.mean(armse), np.mean(awmape)))


if __name__ == "__main__":
    main()
