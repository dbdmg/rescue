import argparse
from run_single import train as single_train, get_args
from run_double import train as double_train
from neural_net.cnn_configurations import validation_dict
import sys

losses = ['bcemse', 'dicemse', 'bdmse', 'bsmse', 'siousiou', 'sioumse', 'bcef1mse']

def main():
    if len(sys.argv) > 1:
        ls = int(sys.argv[1]) 
        del sys.argv[1]
    else:
        ls = None

    args = get_args()
    for seed in [1,2,3]:
        for k in list(validation_dict.keys())[::-1]:
#             for model in ['unet', 'segnet', 'nestedunet', 'attentionunet']:
            for model in ['attentionunet']:

                args.model_name = model
                args.seed = seed
                args.key = k
                args.losses = losses[ls] if ls is not None else None
                print(f'>> run_single {" ".join([f"--{k}={v}" for k, v in vars(args).items()])}\n')
#                 double_train(args)
                single_train(args)
                print("\n\n\n")
          
if __name__ == '__main__':
    main()
    
