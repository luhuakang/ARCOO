from core.data import DesignBenchTask, TaskDataset
from core.logger import Logger as Logger
from core.utils import RiskSuppressionFactor, sample_langevin
from core.arcoo.trainers import DualHeadSurogateTrainer
from core.arcoo.nets import DualHeadSurogateModel
from core.arcoo.optimizer import Optimizer
import numpy as np
import os
import json
import torch


def arcoo(config):
    logger = Logger(config['log_dir'])
    with open(os.path.join(config['log_dir'], "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    task = DesignBenchTask(config['task'], relabel=config['task_relabel'])
    if config['normalize_ys']:
        task.map_normalize_y()
    if task.is_discrete:
        task.map_to_logits()
    if config['normalize_xs']:
        task.map_normalize_x()

    x = torch.Tensor(task.x).cuda()
    y = torch.Tensor(task.y).cuda()
    
    # build the dual-head surrogate model
    dhs_model = DualHeadSurogateModel(np.prod(x.shape[1:]),
                                    config['surrogate_hidden'],
                                    np.prod(y.shape[1:])).cuda()
    print(dhs_model)
    
    init_m = config['init_m'] * np.sqrt(np.prod(x.shape[1:]))
    trainer = DualHeadSurogateTrainer(dhs_model,
                                    dhs_model_prediction_opt=torch.optim.Adam, dhs_model_energy_opt=torch.optim.Adam, 
                                    surrogate_lr=0.001, init_m=init_m,
                                    ldk=config['Ld_K'])

    # create data loaders
    dataset = TaskDataset(x, y)
    train_dataset_size = int(len(dataset) * (1 - config['val_size']))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                            [train_dataset_size, (len(dataset)- train_dataset_size)])
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=config['train_batch'],
                                        shuffle=True)
    validate_dl = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=config['train_batch'],
                                            shuffle=True)
    # train the surrogate model
    trainer.launch(train_dl, validate_dl,
                logger, config['train_epoch'], config['e_train'])

    torch.save(dhs_model.state_dict(), os.path.join(config['log_dir'], 'model_para.pkl'))
    if config['save_model']:
        torch.save(dhs_model.state_dict(), os.path.join('./config/arcoo', config['log_dir'].split('/')[-3], 'model_para.pkl'))

    # select the top k initial designs from the dataset
    indice = torch.topk(y[:, 0], config["online_solutions_batch"])[1].unsqueeze(1)
    init_xt = x[indice].squeeze(1)
    init_yt = y[indice].squeeze(1)

    # get energy scalar
    energy_min = dhs_model(init_xt)[1].mean().detach().cpu().numpy()
    energy_max = dhs_model(sample_langevin(init_xt, dhs_model,
                                            stepsize=init_m,
                                            n_steps=config['Ld_K_max'],
                                            noise=False
                                            ))[1].mean().detach().cpu().numpy()
    uc = RiskSuppressionFactor(energy_min, energy_max, init_m = init_m)
    
    optimizer = Optimizer(config['opt_config'], logger, task,
                        trainer, init_xt, init_yt,
                        dhs_model=dhs_model)

    optimizer.optimize(uc)

if __name__ == '__main__':
    pass
