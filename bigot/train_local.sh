
echo "Running BIGOT locally"

python -m  trainer.task \
  --source-path data/nytfb/triples \
  --target-path data/nytfb/sentences \
  --dataset nytfb
