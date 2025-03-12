#!/bin/bash

MODELS_DIR="models"
CSV_FILE="models/results3.csv"
ITERATIONS=(15)  # Example iterations
EPSILONS=(0.0001 0.001 0.01)  # Example epsilon values
EPISODES=10  # Number of evaluation episodes

mkdir -p $MODELS_DIR

# Train original PPO
for ITER in "${ITERATIONS[@]}"; do
    python main.py --iterations $ITER
    MODEL_NAME="ppo_actor_ver1_${ITER}"
    python main.py --mode test --actor_model "$MODELS_DIR/${MODEL_NAME}.pth" --episodes $EPISODES
done

# Train modified PPO with different epsilons
for ITER in "${ITERATIONS[@]}"; do
    for EPS in "${EPSILONS[@]}"; do
        python main2.py --iterations $ITER --epsilon $EPS
        MODEL_NAME="ppo_actor_ver2_${ITER}_${EPS}"
        python main2.py --mode test --actor_model "$MODELS_DIR/${MODEL_NAME}.pth" --episodes $EPISODES
    done
done

# Generate CSV
echo "version,num_iter,epsilon,training_time,average_score,all_scores" > $CSV_FILE

for FILE in $MODELS_DIR/*.txt; do
    BASENAME=$(basename "$FILE")
    
    if [[ "$BASENAME" =~ ppo_actor_ver1_([0-9]+).txt$ ]]; then
        VERSION="ver1"
        NUM_ITER="${BASH_REMATCH[1]}"
        EPSILON="N/A"
        TRAINING_TIME=$(cat "$FILE")
        EVAL_FILE="${FILE%.txt}_eval.txt"
    elif [[ "$BASENAME" =~ ppo_actor_ver2_([0-9]+)_([0-9\.]+).txt$ ]]; then
        VERSION="ver2"
        NUM_ITER="${BASH_REMATCH[1]}"
        EPSILON="${BASH_REMATCH[2]}"
        TRAINING_TIME=$(cat "$FILE")
        EVAL_FILE="${FILE%.txt}_eval.txt"
    else
        continue
    fi
    
    if [[ -f "$EVAL_FILE" ]]; then
        SCORES=$(cat "$EVAL_FILE" | tr '\n' ',')
        AVG_SCORE=$(awk -v FS="," '{sum=0; n=0; for(i=1; i<=NF; i++) {sum+=$i; n++}} END {if(n>0) print sum/n; else print "N/A"}' <<< "$SCORES")
    else
        SCORES="N/A"
        AVG_SCORE="N/A"
    fi
    
    echo "$VERSION,$NUM_ITER,$EPSILON,$TRAINING_TIME,$AVG_SCORE,$SCORES" >> $CSV_FILE
done

echo "Experiment completed! Results saved to $CSV_FILE"
