import argparse
def parameters():
    parser = argparse.ArgumentParser(description='train the model')
    parser.add_argument('--type', metavar='type', type=str,
                        help='model type', required=True)
    parser.add_argument('--batch', metavar='batch', type=int,
                        help='batch size', default=32, required=False)
    parser.add_argument('--lr', metavar='lr', type=float,
                        help='learning rate', default=0.001, required=False)
    parser.add_argument('--epochs', metavar='epochs', type=int,
                        help='number of epochs', default=500, required=False)
    args = parser.parse_args()
    return args