import json
import os
import re
import os.path as path

results = {}
models = ['darts', 'darts-lfm', 'pcdarts', 'pcdarts-lfm', 'pdarts', 'pdarts-lfm']
for model in models:
    results[model] = {}
    for f in os.listdir(model):
        if 'search-EXP' in f:
            results[model][f] = {'args': None, 'param_size': None,
                                 'lr': [], 'lr_rw': [], 'genotype': [], 'beta': [],
                                 'train_acc': [], 'train_loss': [], 'train_loss_rw': [],
                                 'val_acc': None, 'val_loss': None}
            with open(path.join(model, f, 'log.txt')) as search_log:
                for line in search_log:
                    if 'args' in line:
                        results[model][f]['args'] = line
                    elif 'param size' in line:
                        results[model][f]['param_size'] = line
                    elif 'lr' in line:
                        start = line.find('lr') + 2
                        if 'lr_rw' in line:
                            end = line.find('lr_rw')
                        else:
                            end = -1
                        results[model][f]['lr'].append(float(line[start: end]))
                    elif 'lr_rw' in line:
                        start = line.find('lr_rw') + len('lr_rw ')
                        results[model][f]['lr_rw'].append(float(line[start:]))
                    elif 'genotype = ' in line:
                        start = line.find('genotype = ') + len('genotype = ')
                        results[model][f]['genotype'].append(line[start:])
                    elif 'beta = ' in line:
                        start = line.find('beta = ') + len('beta = ')
                        results[model][f]['beta'].append(line[start:])
                    elif