

### How to run

#### NB: About paths, hydra and mlflow

The code will use mlflow to log experiments and hydra to manage configurations.
There will be different relevant paths, by default all in the HOME folder:

- mlflow database: `mlflow.db` (sqlite file with experiment tracking info)
- mlflow artifacts: `mlflow-artifacts/` (folder with artifacts/models logged by mlflow)
- hydra multirun folder: `multirun/` (folder with hydra multirun outputs, including slurm logs etc)
- datasets cache folder: `datasets/` (folder where datasets are downloaded and cached)
- hydra non-multi run folder: `outputs/` (folder for hydra without `-m`, not used much here)


#### Run some experiments

```bash
cd .....


# trigger installation
uv venv
uv run src/train_toy2d.py --help


# activate the virtual environment, to used directly python without uv
source .venv/bin/activate
# for autocompletion with python ...
eval "$(python src/train_toy2d.py -sc install=bash)"


# Run locally or on a compute node after manual login
#python src/train_toy2d.py +light=train_light_toy2d
# or as multirun to avoid having both outputs/ and multirun/
python src/train_toy2d.py -m +light=train_light_toy2d
m

# or to schedule with slurm (from the frontend node)
# - need "-m" for multirun, which is needed for the slurm launcher, even for a single run
# - IMPORTANT: need to pip install so the slurm launcher finds the package correctly
uv pip install -e .
python src/train_toy2d.py -m +light=train_light_toy2d +slurm=cpu
# or for JZ
python src/train_toy2d.py -m +light=train_light_toy2d +slurm=jzv100

## ALL
uv pip install -e .
python src/train_toy2d.py   -m +light=train_light_toy2d   +slurm=gpu24
python src/train_cifar10.py -m +light=train_light_cifar10 +slurm=gpu24



# Remove the light=train_light_toy2d to run the full config
uv pip install -e .
python src/train_toy2d.py -m +slurm=cpu
python src/train_toy2d.py -m +slurm=jzv100
#...

# Some watching
watch -n 3 bash -c '"squeue -u $USER; echo .. ; ls -ltr .. | tail -4 ; echo ./ ; ls -ltr ./ | tail -4"'
```



#### Look at MLflow server ui (with back tunneling)

Connect and tunnel:

```bash
# IF YOU HAVE A CONTROL CONNECTION, start the tunnel with:
ssh -O forward -L 3839:localhost:31333 joecluster
ssh joecluster

# IF NOT, CONNECT AND TUNNEL
ssh -L 3839:localhost:31333 joecluster
```

Schedule, back tunnel and start the mlflow server on the cluster (to avoid loading the frontend node):

```bash
tmux a
srun -p LONG -t 6-00:00:00  -c 2 --mem=10G  --pty bash -i
# THEN
ssh -NT -i ~/.ssh/id_ed25519_betweencluster -oExitOnForwardFailure=yes -R 31333:localhost:3999 calcul-slurm-lahc-2 &
trap "kill $!" EXIT
cd closedformfm
. .venv/bin/activate
fg # for the password...
# Ctrl+Z, bg
mlflow server --backend-store-uri sqlite:///$HOME/mlflow.db --default-artifact-root $HOME/mlflow-artifacts --port 3999
```

http://localhost:3839

----

#### OLD RANDOM

```bash
ssh -L 3839:localhost:31333 labslurm

tmux a
srun -p LONG -t 6-00:00:00  -c 2 --mem=10G  --pty bash -i

ssh -NT -i ~/.ssh/id_ed25519_betweencluster -oExitOnForwardFailure=yes -R 31333:localhost:3999 calcul-slurm-lahc-2 &
trap "kill $!" EXIT
cd closedformfm
. .venv/bin/activate
fg # for the password...
# Ctrl+Z, bg
mlflow server --backend-store-uri sqlite:///$HOME/mlflow.db --default-artifact-root $HOME/mlruns --port 3999


http://locatlhost:3839
```




### Clean up notes

- [x] use uv
- [x] clarify what is best practice for wconf and hydra, decide folder structure
- [x] import one learning, typically 2D toy data
- [x] example slurm conf etc
- [ ] import a figure code, maybe the histogram one, for 2d toy data
- [ ] import the rest
- [ ] add license
- [ ] add tests

