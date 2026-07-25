#!/usr/bin/env bash
# One vLLM server per visible GPU + driver over the cell in repro/config.json.
set -uo pipefail
echo "[run] start $(date -u +%FT%TZ)"
nvidia-smi -L || true
NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
echo "[run] $NGPU GPUs"
WORK=/work
mkdir -p "$WORK"
export HF_HOME="${HF_HOME:-$WORK/hf}"

pip install -q bm25s PyStemmer datasets pyarrow 2>&1 | tail -2

MODEL=$(python3 -c "import json;print(json.load(open('repro/config.json'))['model'])")
MAXLEN=$(python3 -c "import json;print(json.load(open('repro/config.json'))['max_model_len'])")

echo "[run] preparing data"
python3 repro/prepare_data.py --workdir "$WORK" || { echo "[run] DATA PREP FAILED"; exit 1; }

echo "[run] downloading model $MODEL"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" \
  || { echo "[run] MODEL DOWNLOAD FAILED"; exit 1; }

PORTS=""
for i in $(seq 0 $((NGPU-1))); do
  PORT=$((8100+i))
  echo "[run] starting vLLM on GPU $i port $PORT"
  CUDA_VISIBLE_DEVICES=$i python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --max-model-len "$MAXLEN" \
    --gpu-memory-utilization 0.90 --max-num-seqs 16 \
    > "$WORK/vllm_$i.log" 2>&1 &
  PORTS="$PORTS $PORT"
done

for PORT in $PORTS; do
  ok=""
  for t in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then ok=1; break; fi
    sleep 10
  done
  if [ -z "$ok" ]; then
    echo "[run] SERVER $PORT FAILED TO START; last vllm logs:"
    tail -80 "$WORK"/vllm_*.log
    exit 1
  fi
  echo "[run] server $PORT healthy"
done

python3 repro/driver.py --config repro/config.json --workdir "$WORK" --ports "$PORTS"
RC=$?
echo "[run] driver exit code $RC"
if [ $RC -ne 0 ]; then tail -40 "$WORK"/vllm_*.log; fi
exit $RC
