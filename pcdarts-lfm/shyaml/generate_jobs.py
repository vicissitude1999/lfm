import random

datasets = ['cifar100', 'cifar10']
seeds = [1,2,3]
counter = 1

for i, dataset in enumerate(datasets):
    for j, seed in enumerate(seeds):
        job_name = "renyi-pcdarts-{}-{}".format(dataset, str(seed))
        index = random.randint(1, 100)
        with open('search-{}.sh'.format(str(counter)), 'w') as sh:
            with open('search.sh', 'r') as sh_template:
                sh.write(sh_template.read().format(dataset, str(seed)))
        with open('search-{}.yaml'.format(str(counter)), 'w') as yaml:
            with open('search.yaml', 'r') as yaml_template:
                yaml.write(yaml_template.read().format(job_name, str(counter)))
        counter += 1