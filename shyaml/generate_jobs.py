import sys
sys.path.append("..")
import genotypes
from genotypes import arch_names

datasets = ['cifar100', 'cifar10']
counter = 1

for k in arch_names:
    dataset = k.split('_')[2]
    job_name = "renyi-{}".format(k)
    with open('train-{}.sh'.format(str(counter)), 'w') as sh:
        with open('train.sh', 'r') as sh_template:
            sh.write(sh_template.read().format(dataset, k, arch_names[k]))
    with open('train-{}.yaml'.format(str(counter)), 'w') as yaml:
        with open('train.yaml', 'r') as yaml_template:
            yaml.write(yaml_template.read().format(k, str(counter)))
    counter += 1