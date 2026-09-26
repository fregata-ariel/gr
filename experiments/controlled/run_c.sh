#!/bin/bash
# C: train per source bundle on Colab session "c", then re-score each model on its OOD targets.
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
C=$S/c; SES=c2
timeout 300 colab upload -s $SES training/train_ar.py /content/train_ar.py 2>&1 | tail -1
timeout 300 colab upload -s $SES training/grammar_mask.py /content/grammar_mask.py 2>&1 | tail -1
declare -A BUNDLE=( [layered]=tok_c_layered [structured]=tok_c_structured [depth1]=tok_c_depth_id [merge2]=tok_c_merge_id [balanced]=tok_c_balanced )
declare -A CFGS=( [layered]="base mask ptr" [structured]="base mask" [depth1]="base mask" [merge2]="base mask" [balanced]="base mask" )
declare -A TGTS=( [layered]="tok_c_lay2str tok_c_lay2spa" [structured]="tok_c_str2lay tok_c_str2spa" [depth1]="tok_c_depth" [merge2]="tok_c_merge" [balanced]="" )
for src in layered structured depth1 merge2 balanced; do
  B=${BUNDLE[$src]}
  timeout 300 colab upload -s $SES training/train_ar.py /content/train_ar.py 2>&1 | tail -1
  timeout 300 colab upload -s $SES training/grammar_mask.py /content/grammar_mask.py 2>&1 | tail -1
  for f in train.jsonl val.jsonl test.jsonl vocab.json meta.json; do timeout 300 colab upload -s $SES data/$B/$f /content/$f 2>&1 | tail -1; done
  timeout 300 colab upload -s $SES data/$B/vocab.json /content/vocab_$src.json 2>&1 | tail -1
  timeout 300 colab upload -s $SES data/$B/meta.json /content/meta_$src.json 2>&1 | tail -1
  for cfg in ${CFGS[$src]}; do for seed in 0 1 2; do
    NAME=c_${src}_${cfg}_n24_s${seed}
    if [ -f runs/$NAME/test_scores.jsonl ]; then echo "skip $NAME"; continue; fi
    echo "=== $NAME start $(date +%T) ==="
    timeout 6000 colab exec -s $SES -f $C/w/w_$NAME.py --timeout 5400 2>&1 | grep -E "epoch   1 |early stop|test NLL|wrote|Error|error|Traceback" | tail -8
    mkdir -p runs/$NAME
    for f in samples.json history.json model.pt val_scores.jsonl test_scores.jsonl samples_constrained.json; do
      timeout 300 colab download -s $SES /content/$NAME/$f runs/$NAME/$f 2>&1 | grep -iE "error|not found"
    done
    uv run python -m training.eval_samples --samples runs/$NAME/samples.json --vocab data/$B/vocab.json --train-tokens data/$B/train.jsonl --out runs/$NAME/eval.json 2>&1 | grep -E "well_formed_rate"
    echo "=== $NAME done $(date +%T) ==="
  done; done
  # re-score this source's models on its OOD targets (skip targets already scored locally)
  jobs="["
  for tgt in ${TGTS[$src]}; do
    missing=0
    for cfg in ${CFGS[$src]}; do for seed in 0 1 2; do
      [ -f runs/c_${src}2${tgt#tok_c_}_${cfg}_n24_s${seed}/test_scores.jsonl ] || missing=1
    done; done
    if [ $missing = 0 ]; then echo "skip rescore $src -> $tgt"; continue; fi
    timeout 300 colab upload -s $SES data/$tgt/test.jsonl /content/test_$tgt.jsonl 2>&1 | tail -1
    for cfg in ${CFGS[$src]}; do for seed in 0 1 2; do
      NAME=c_${src}_${cfg}_n24_s${seed}
      jobs="$jobs{\"run\":\"$NAME\",\"model\":\"/content/${NAME}_model.pt\",\"config\":\"/content/${NAME}_samples.json\",\"vocab\":\"/content/vocab_$src.json\",\"meta\":\"/content/meta_$src.json\",\"test\":\"/content/test_$tgt.jsonl\",\"out\":\"/content/${NAME}__${tgt}.jsonl\"},"
    done; done
  done
  # checkpoints from local runs/ (flat paths: VM state may have been lost)
  [ "$jobs" != "[" ] && for cfg in ${CFGS[$src]}; do for seed in 0 1 2; do
    NAME=c_${src}_${cfg}_n24_s${seed}
    timeout 300 colab upload -s $SES runs/$NAME/model.pt /content/${NAME}_model.pt 2>&1 | tail -1
    timeout 300 colab upload -s $SES runs/$NAME/samples.json /content/${NAME}_samples.json 2>&1 | tail -1
  done; done
  if [ "$jobs" != "[" ]; then
    echo "${jobs%,}]" > $C/rescore_jobs_$src.json
    timeout 300 colab upload -s $SES $C/rescore_jobs_$src.json /content/rescore_jobs.json 2>&1 | tail -1
    echo "=== rescore $src $(date +%T) ==="
    timeout 2400 colab exec -s $SES -f $S/rescore_c.py --timeout 1800 2>&1 | grep -E "RESCORED|RESCORE-DONE|Error|error|Traceback"
    for tgt in ${TGTS[$src]}; do for cfg in ${CFGS[$src]}; do for seed in 0 1 2; do
      NAME=c_${src}_${cfg}_n24_s${seed}; OUT=runs/c_${src}2${tgt#tok_c_}_${cfg}_n24_s${seed}
      mkdir -p $OUT
      timeout 300 colab download -s $SES /content/${NAME}__${tgt}.jsonl $OUT/test_scores.jsonl 2>&1 | grep -iE "error|not found"
      cp runs/$NAME/eval.json $OUT/eval.json 2>/dev/null
    done; done; done
  fi
done
echo RUN-C-DONE
