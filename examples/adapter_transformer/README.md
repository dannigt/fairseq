### Adversarial language classifier

#### Introduction
These parameters should be set differently depending on the training data: 
* `--num-language-to-classify`: total number of languages
* `--language-classifier-one-vs-rest`: if non-zero, language classification will be binary (this class vs the rest) 
* `--actual-vocab-size`: vocabulary size, excluding language tags

Before training, please replace `trainer.py` by `trainer_adv.py`.
The differences to `train.py` are:
* It builds a separate optimizer for the adversarial classifier.
* It scales num_updates, so that the optimizer schedules are unaffected by the additional steps from the alternating training.

The following registered components are implemented in this user dir under `adapter_transformer_src`:
```
--arch language_classification_transformer
--task multilingual_translation_adversarial_language_classifier
--criterion language_classification_cross_entropy
```


#### Training on English-centric data
```bash
# English-centric example
lang_pairs="en-ms,ms-en,en-tl,tl-en,en-jv,jv-en"
LANS="en jv ms tl"

# Having all directions in dev set
for slan in $LANS; do
        for tlan in $LANS; do
                if [[ $slan != $tlan ]]; then
                        if [ -z "$valid_lang_pairs" ]; then
                                valid_lang_pairs=$slan"-"$tlan
                        else
                                valid_lang_pairs=$valid_lang_pairs","$slan"-"$tlan
                        fi
                fi
        done
done

# Effective batch size 16384
bsz=4096
GPUS="0,1"
updates=2

scaling=1   # scaling factor on gradient reversal layer
alt_updates=2   # alternating between classification and translation

path_2_data=""  # path to binarized data
lang_list="" # e.g. /home/dliu/data/wmt21/small_task2/data/prepro_fairseq/lang_list.txt
model="" # name of current run

OUT_DIR="" # output directory, e.g. /home/dliu/data/wmt21/small_task2/model
FAIRSEQ_DIR="" # e.g. /home/dliu/src/fairseq-dev

mkdir -p $OUT_DIR/$model

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train $path_2_data \
  --user-dir $FAIRSEQ_DIR/examples/adapter_transformer/adapter_transformer_src \
  --encoder-normalize-before --decoder-normalize-before \
  --share-all-embeddings \
  --encoder-layers 6 --decoder-layers 6 \ 
  --encoder-attention-heads 16 --decoder-attention-heads 16 \
  --encoder-embed-dim 512 --decoder-embed-dim 512 \
  --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
  --arch language_classification_transformer \
  --task multilingual_translation_adversarial_language_classifier \
  --sampling-method "temperature" \
  --sampling-temperature 5.0 \
  --max-tokens $bsz \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --valid-lang-pairs "$valid_lang_pairs" \
  --criterion language_classification_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 1e-04 --warmup-updates 2500 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --update-freq $updates \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --num-language-to-classify 124 --report-accuracy \
  --language-classifier-one-vs-rest 24 \
  --language-classifier-steps $alt_updates \
  --ddp-backend=legacy_ddp \
  --seed 222 --log-format simple --log-interval 1000 --fp16 \
  --left-pad-source False \
  --grad-reversal-scaling-factor $scaling \
  --actual-vocab-size 165034 \
  --finetune-from-model $OUT_DIR/mm100_small.trimmed/model.pt \
  --save-dir $OUT_DIR/$model 2>&1 | tee $OUT_DIR/$model/train.log 
```
#### Training on Indonesian-centric data

These params are to be set differently from above:
```bash
lang_pairs="id-ms,ms-id,id-tl,tl-id,id-jv,jv-id"
LANS="id jv ms tl"

--language-classifier-one-vs-rest 43
```

#### Inference

```bash
GPU=0
BASEDIR="" #e.g. /home/dliu/data/wmt21/small_task2
name=$1 # model name
FAIRSEQ_MODEL=$BASEDIR/model/$name

mkdir -p $BASEDIR/data/$name/

for sl in $LANS; do
for tl in $LANS; do
        if [[ $sl != $tl ]]; then
          
        CUDA_VISIBLE_DEVICES=$GPU fairseq-generate \
            --user-dir examples/adapter_transformer/adapter_transformer_src \
            $path_2_data \
            --batch-size 256 \
            --remove-bpe 'sentencepiece' \
            --task translation_multi_simple_epoch \
            --path $FAIRSEQ_MODEL/checkpoint_best.pt \
            --fixed-dictionary $FAIRSEQ_MODEL/dict.txt \
            -s $sl -t $tl \
            --beam 5 \ 
            --lang-dict /home/dliu/data/wmt21/small_task2/data/prepro_fairseq/lang_list.txt \
            --lang-pairs $FAIRSEQ_MODEL/language_pairs.txt \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test \
            --dataset-impl mmap \
            --left-pad-source False \
            --distributed-world-size 1 --distributed-no-spawn --fp16 > $BASEDIR/data/$name/$sl-$tl.pred.log

        grep ^H $BASEDIR/data/$name/$sl-$tl.pred.log | cut -f3- > $BASEDIR/data/$name/$sl-$tl.sys
        grep ^T $BASEDIR/data/$name/$sl-$tl.pred.log | cut -f2- > $BASEDIR/data/$name/$sl-$tl.ref

        cat $BASEDIR/data/$name/$sl-$tl.sys | sacrebleu $BASEDIR/data/$name/$sl-$tl.ref -tok spm > $BASEDIR/data/$name/$sl-$tl.res

        fi
done
done

```