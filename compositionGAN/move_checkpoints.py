import os
import argparse
import shutil
# move from thing to thing

def test_checkpoints(root, epoch, dest):
    for f in os.listdir(root):
        f_split = f.split('_')

        if f_split[0] == epoch:
            f_new = 'best_'+'_'.join(f_split[1:])
            shutil.copy(root+f, dest+f_new)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Move and rename training epochs")

    # parser
    parser.add_argument("-i", dest="root",
        help="root")
    parser.add_argument("-e", dest="epoch",
        help="epoch")
    parser.add_argument("-d", dest="dest",
        help="dest")
        
    args = parser.parse_args()
    
    test_checkpoints(args.root, args.epoch, args.dest)