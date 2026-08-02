# LANGUAGE AGNOSTIC VULNERABILITY DETECTION



## Inference

```bash
python3 main.py \
    --dataset 'himanshu17HF/BigVul-Filtered' \
    --source-col 'func_before' \
    --label-col 'vul' \
    --output-dir '/content/drive/MyDrive/results' \
    --max-iter 1000 \
    --random-state 42 \
    --max-length 512 \
    --batch-size 64
    
```