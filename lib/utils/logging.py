import os
import logging
import time
from pathlib import Path

def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.log'.format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, 'log', log_file)
    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    # if len(logging.getLogger('').handlers) < 2:
    logging.getLogger('').addHandler(console)

    return logger, time_str

def create_checkpoint(cfg, phase='train', output_dir=''):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if (output_dir != '') and os.path.exists(output_dir):
        final_output_dir = Path(output_dir)
    else:
        # set up logger
        if not root_output_dir.exists():
            print('=> creating {}'.format(root_output_dir))
            root_output_dir.mkdir()

        # time str
        time_str = time.strftime('%y_%m_%d-%H_%M')

        dataset = cfg.DATASET.DATASET
        dataset = dataset.replace(':', '_')
        cfg_name = f"{cfg.CFG_NAME}-{phase}-{time_str}"

        final_output_dir = root_output_dir / dataset / cfg_name

        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = os.path.join(final_output_dir, 'log')
    if not os.path.exists(log_dir):
        print('=> creating log dir'.format(log_dir))
        os.makedirs(log_dir)

    if phase == 'train':
        tensorboard_log_dir = os.path.join(final_output_dir, 'tblog')
        model_dir = os.path.join(final_output_dir, 'model')
        src_dir = os.path.join(final_output_dir, 'src')
        if not os.path.exists(tensorboard_log_dir): os.makedirs(tensorboard_log_dir)
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        if not os.path.exists(src_dir): os.makedirs(src_dir)
        print('=> creating {}'.format(tensorboard_log_dir))
        print('=> creating {}'.format(model_dir))
        print('=> creating {}'.format(src_dir))

    return str(final_output_dir)
