import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--createpatches", required=False, type=str2bool,
        default="False",help="boolean for whether to create the patches images dataset")
    ap.add_argument("-p", "--processpatches", required=False, type=str2bool,
        default="False",help="boolean for whether to process the patches")
    ap.add_argument("-b", "--batchsize", required=False, type=int,
                    default=64, help="batch size for dataloader")
    ap.add_argument("-s", "--splitting", nargs='+', required=False, type=float,
                    default=[4,1,1], help="list of length 3 [size_train, size_val, size_test]. "
                                        "If summing up to 1, the data will be split randomly across each site,"
                                        "if summing up to 6, the data data will be split by site")
    return ap.parse_args()