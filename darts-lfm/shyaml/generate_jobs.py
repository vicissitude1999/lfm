
datasets = ['cifar100', 'cifar10']
seeds = [1,2,3]

for i, dataset in enumerate(datasets):
    for j, seed in enumerate(seeds):
        job_name = "renyi-darts-{}-{}".format(dataset, str(seed))
        with open('search-{}.sh'.format(str(i+j)), 'w') as sh:
            with open('search.sh', 'r') as sh_template:
                sh.write(sh_template.read().format(dataset, str(seed)))
        with open('search-{}.yaml'.format(str(i+j)), 'w') as yaml:
            with open('search.yaml', 'r') as yaml_template:
                yaml.write(yaml_template.read().format(job_name))