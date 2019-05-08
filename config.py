import tensorflow as tf

class AgentConfig(object):
    
    train = False

    # LSTM parameters
    h_sz = 50
    lstm_input = 32

    # test model
    load_model = 'logs/models/'
    save_res = True

    # train model
    save_model = 'logs/models/'
    logs = 'logs/logs/'
    """
    memory_sz = 500000
    learn_start = 5000
    test_sp = 1000
    save_sp = 50000
    max_sp = 1000000
    target_q_update_sp = 10000
    """
    #debug
    memory_sz = 500
    learn_start = 50
    test_sp = 100
    save_sp = 500
    max_sp = 1000
    target_q_update_sp = 100

    batch_sz = 32
    train_freq = 4
    discount = 0.99
    # learning rate
    lr = 0.0001
    lr_minimum = 0.000025
    lr_decay = 0.5
    lr_decay_sp = 1000000
    # experience replay
    ep_start = 1.  # 1: fully random; 0: no random
    ep_end = 0.1
    ep_end_t = 1000000

class EnvironmentConfig(object):
    # params for environment
    rows  = 63
    cols = 63
    channels = 3
    test_batch = 1024  # test how many patches at a time
    stop_sp = 3
    reward_cal = 'psnr' # psnr, ssim, mse, nrmse
    reward_func = 'sp_reward'

    # data path
    train_dir = 'data/train/'
    val_dir = 'data/valid/'
    dataset = '/usr/local/home/ssbw5/adp/FinalProject/data/test_images/moderate'
    new_image = False


class ModelConfig(AgentConfig, EnvironmentConfig):
    pass


def get_config(ARGS):
    config = ModelConfig
    
    for k in ARGS:
        v = ARGS[k].value
        if hasattr(config, k):
            setattr(config, k, v)

    return config
