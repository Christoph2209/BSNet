import torch
import numpy as np
import os
import time

# Assuming `model` is already defined and loaded, and you want to save it after evaluation:

if __name__ == '__main__':
    params = parse_args('test')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)

    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    feature_model = backbone.Conv4NP

    # Model setup based on the method specified
    if params.method in ['relationnet']:
        model = RelationNet(feature_model, loss_type='mse', **few_shot_params)
    elif params.method in ['OurNet']:
        model = OurNet(feature_model, loss_type='mse', **few_shot_params)
    elif params.method in ['CosineBatch']:
        model = CosineBatch(feature_model, loss_type='mse', **few_shot_params)
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    # Load the model checkpoint
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    print('checkpoint_dir:', checkpoint_dir)

    # Load the best model from checkpoint
    modelfile = get_best_file(checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    # Evaluate the model and calculate accuracy
    novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"), "novel_data_file.hdf5")
    cl_data_file = feat_loader.init_loader(novel_file)

    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query=15, **few_shot_params)
        acc_all.append(acc)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

    # Save the model state after evaluation
    model_save_path = f"./model_saved_{params.method}_{params.dataset}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    # Save evaluation results
    with open('./BSNet/record/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' % (
            params.dataset, "novel", params.model, params.method, aug_str, params.n_shot, params.train_n_way, params.test_n_way)
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        f.write('Time: %s, Setting: %s, Acc: %s \n' % (timestamp, exp_setting, acc_str))
