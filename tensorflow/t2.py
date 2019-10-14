import argparse

def main(args):
    print(args.num_steps)
    print(args.batch_size)
    print(args.log_every)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())