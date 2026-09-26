#!/bin/bash
# Milestone sweep runner (idempotent): per size and source, train mask x 3 seeds, then re-score on the targets.
cd /home/fischeri/Projects/Compiler/gr
S=/tmp/claude-1000/-home-fischeri-Projects-Compiler-gr/b84df06a-6417-4e37-b3a6-966467410dd9/scratchpad
W=$S/sw/w; SES=${SWEEP_SESSION:-sw}
ok() { colab sessions 2>&1 | grep -q "^\[$SES\]"; }
timeout 300 colab upload -s $SES training/train_ar.py /content/train_ar.py 2>&1 | tail -1
timeout 300 colab upload -s $SES training/grammar_mask.py /content/grammar_mask.py 2>&1 | tail -1
for n in 12 16 24 32 48; do
  for src in lay mix; do
    B=tok_s${n}_${src}
    case $src in lay) TG="tok_s${n}_lay2str tok_s${n}_lay2spa";; mix) TG="tok_s${n}_mix2str tok_s${n}_mix2spa tok_s${n}_mix2lay";; esac
    need=0; for seed in 0 1 2; do [ -f runs/c_s${n}_${src}_mask_n${n}_s${seed}/test_scores.jsonl ] || need=1; done
    for tgt in $TG; do for seed in 0 1 2; do [ -f runs/c_s${n}_${src}2${tgt#tok_s${n}_${src}2}_mask_n${n}_s${seed}/test_scores.jsonl ] || need=1; done; done
    if [ $need = 0 ]; then echo "skip n$n $src (complete)"; continue; fi
    ok || { echo "SESSION-LOST before n$n $src"; exit 3; }
    for f in train.jsonl val.jsonl test.jsonl vocab.json meta.json; do timeout 300 colab upload -s $SES data/$B/$f /content/$f 2>&1 | tail -1; done
    timeout 300 colab upload -s $SES data/$B/vocab.json /content/vocab_${n}_${src}.json 2>&1 | tail -1
    timeout 300 colab upload -s $SES data/$B/meta.json /content/meta_${n}_${src}.json 2>&1 | tail -1
    for seed in 0 1 2; do
      NAME=c_s${n}_${src}_mask_n${n}_s${seed}
      if [ -f runs/$NAME/test_scores.jsonl ]; then echo "skip $NAME"; continue; fi
      ok || { echo "SESSION-LOST at $NAME"; exit 3; }
      echo "=== $NAME start $(date +%T) ==="
      timeout 6000 colab exec -s $SES -f $W/w_$NAME.py --timeout 5400 2>&1 | grep -E "early stop|test NLL|Error|Traceback|lost" | tail -4
      mkdir -p runs/$NAME
      for f in samples.json history.json model.pt val_scores.jsonl test_scores.jsonl samples_constrained.json; do
        timeout 300 colab download -s $SES /content/$NAME/$f runs/$NAME/$f 2>&1 | grep -iE "error|not found"
      done
      if [ -f runs/$NAME/test_scores.jsonl ]; then
        uv run python -m training.eval_samples --samples runs/$NAME/samples.json --vocab data/$B/vocab.json --train-tokens data/$B/train.jsonl --out runs/$NAME/eval.json 2>&1 | grep -E "well_formed_rate"
        echo "=== $NAME done $(date +%T) ==="
      else
        echo "=== $NAME FAILED $(date +%T) ==="; rm -rf runs/$NAME; exit 3
      fi
    done
    # re-scoring on targets (flat checkpoints from local runs/)
    jobs="["
    for tgt in $TG; do
      short=${tgt#tok_s${n}_${src}2}
      missing=0; for seed in 0 1 2; do [ -f runs/c_s${n}_${src}2${short}_mask_n${n}_s${seed}/test_scores.jsonl ] || missing=1; done
      [ $missing = 0 ] && { echo "skip rescore n$n $src -> $short"; continue; }
      timeout 300 colab upload -s $SES data/$tgt/test.jsonl /content/test_$tgt.jsonl 2>&1 | tail -1
      for seed in 0 1 2; do
        NAME=c_s${n}_${src}_mask_n${n}_s${seed}
        jobs="$jobs{\"run\":\"$NAME\",\"model\":\"/content/${NAME}_model.pt\",\"config\":\"/content/${NAME}_samples.json\",\"vocab\":\"/content/vocab_${n}_${src}.json\",\"meta\":\"/content/meta_${n}_${src}.json\",\"test\":\"/content/test_$tgt.jsonl\",\"out\":\"/content/${NAME}__${tgt}.jsonl\"},"
      done
    done
    if [ "$jobs" != "[" ]; then
      for seed in 0 1 2; do NAME=c_s${n}_${src}_mask_n${n}_s${seed}; timeout 300 colab upload -s $SES runs/$NAME/model.pt /content/${NAME}_model.pt 2>&1 | tail -1; timeout 300 colab upload -s $SES runs/$NAME/samples.json /content/${NAME}_samples.json 2>&1 | tail -1; done
      echo "${jobs%,}]" > $S/sw/rescore_jobs_${n}_${src}.json
      timeout 300 colab upload -s $SES $S/sw/rescore_jobs_${n}_${src}.json /content/rescore_jobs.json 2>&1 | tail -1
      echo "=== rescore n$n $src $(date +%T) ==="
      timeout 2400 colab exec -s $SES -f $S/rescore_c.py --timeout 1800 2>&1 | grep -E "RESCORE-DONE|Error|Traceback|lost"
      for tgt in $TG; do short=${tgt#tok_s${n}_${src}2}; for seed in 0 1 2; do
        NAME=c_s${n}_${src}_mask_n${n}_s${seed}; OUT=runs/c_s${n}_${src}2${short}_mask_n${n}_s${seed}; mkdir -p $OUT
        timeout 300 colab download -s $SES /content/${NAME}__${tgt}.jsonl $OUT/test_scores.jsonl 2>&1 | grep -iE "error|not found"
        [ -f $OUT/test_scores.jsonl ] || rmdir $OUT 2>/dev/null
        cp runs/$NAME/eval.json $OUT/eval.json 2>/dev/null
      done; done
    fi
  done
done
echo RUN-SWEEP-DONE
