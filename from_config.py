from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
from comet_ml import Experiment
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = config.trainer.visible_devices

        # create the experiments dirs
        create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

        print('Create the data generator.')
        data_loader = factory.create("data_loader."+config.data_loader.name)(config)

        print('Create the model.')
        model = factory.create("models."+config.model.name)(config)

        print('Create the trainer')
        trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader.get_train_data(), config)

        print('Start training the model.')
        trainer.train()

    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
