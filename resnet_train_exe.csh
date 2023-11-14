#!/bin/csh
#SBATCH --mem=32g
#SBATCH --time=1-0
#SBATCH --mail-type=END
#SBATCH --mail-user=nimrod.kremer
#SBATCH --output=/cs/labs/yweiss/nimrod.kremer/tmp/resnet_fgnet_train.out
#SBATCH --gres=gpu:1,vmem:16gb
#SBATCH --requeue
cd /cs/labs/yweiss/nimrod.kremer/codes/truly_invariant_cnn
source /cs/labs/yweiss/nimrod.kremer/venv/tsinv/bin/activate.csh
python /cs/labs/yweiss/nimrod.kremer/codes/mobileye/fignet_mobileye/train_resnet50.py

