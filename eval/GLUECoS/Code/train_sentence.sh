# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-Sentiment_EN_ES}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
OUT_DIR=${5:-"$REPO/Results"}
TRAIN_DATA=${6:-"train_augmented_prefix-tuning_0.2.txt"}
VAL_DATA=${7:-"validation.txt"}
TEST_DATA=${8:-"test.txt"}

EPOCH=5
BATCH_SIZE=16
MAX_SEQ=256

dir=`basename "$TASK"`
if [ $dir == "Devanagari" ] || [ $dir == "Romanized" ]; then
  OUT=`dirname "$TASK"`
else
  OUT=$TASK
fi

python $PWD/Code/BertSequence.py \
  --data_dir $DATA_DIR/$TASK \
  --output_dir $OUT_DIR/$OUT \
  --train_data_file $TRAIN_DATA \
  --val_data_file $VAL_DATA \
  --test_data_file $TEST_DATA \
  --model_type $MODEL_TYPE \
  --model_name $MODEL \
  --num_train_epochs $EPOCH \
  --train_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ \
  --save_steps -1