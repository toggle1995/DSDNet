# OUTPUT_DIR: path to output dir
OUTPUT_DIR='../workshop/lab_record/logs/dsd/nwpu/'
LOG_DIR='h81_0705_r50_id1_condiff_concat_sturefine_stuout_3pairs_noise0to10_randaug_erase0.15_rmsp_cls10_wd1e-5_lr1e-4_minlr1e-6'

mkdir -p "$OUTPUT_DIR/$LOG_DIR"/saved_code
rsync -av --include='*/' --include='*.py' --include='*.sh' --exclude='*.jpg' --exclude='*.pth' --exclude='*' . "$OUTPUT_DIR/$LOG_DIR"/saved_code/

# DATA_PATH: path to datasets
DATA_PATH='../dataset/nwpu_resisc45/split'

srun --partition=Gveval2-T --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=h81_0705 --kill-on-bad-exit=1 --quotatype=spot --async -o "$OUTPUT_DIR/$LOG_DIR"/log_test.out --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 2 --master_port 18105 train.py --batch_size 64 --base_lr 0.0001 --base_min_lr 1e-6 --wd 1e-5 --root_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR" --optim rmsp \
--total_epoch 405 --train_index 1 --data_root $DATA_PATH --backbone_name res50
