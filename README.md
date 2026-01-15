By using just 2 hours of russian speech:
- [Before fine tune](https://github.com/user-attachments/files/24613013/out.2.wav)
- [After finetune](https://github.com/user-attachments/files/24613434/output.wav)

1. Expand the Tokenizer for Russian
```bash
python expand_tokenizer.py \
    --dataset ./dataset \
    --output ./RussianSoprano \
    --model ekwek/Soprano-1.1-80M
```

2. Generate Audio Tokens 
```bash
python generate_dataset.py --input-dir ./dataset
```

3. Train the Model
```bash
python train.py \
    --input-dir ./dataset \
    --save-dir ./checkpoints/ \
    --model-path ./RussianSoprano/ \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --eval-steps 10 \
    --save-steps 10 \
    --top-k 10
```
