
import argparse
def parameters():
    parser = argparse.ArgumentParser(description='train the model')
    parser.add_argument('--dataset', metavar='dataset', type=str,
                        help='a directory with a dataset', required=False, default='images')
    parser.add_argument('--name', metavar='name', type=str,
                        help='model name', required=True, nargs='+')
    parser.add_argument('--batch', metavar='batch', type=int,
                        help='batch size', default=32, required=False)
    parser.add_argument('--lr', metavar='lr', type=float,
                        help='learning rate', default=0.001, required=False)
    parser.add_argument('--epochs', metavar='epochs', type=int,
                        help='number of epochs', required=False)
    parser.add_argument('--weights', metavar='weights', type=str,
                        help='file with a trained weights', nargs='+', required=False)
    parser.add_argument('--device', metavar='device', type=str,
                        help='cpu or cuda', choices=['cuda', 'cpu'], default='cuda', required=False)
    args = parser.parse_args()
    return args