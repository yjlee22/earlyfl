# Experiment parameters
generators=("roentgen" "sd1.4" "sd1.5" "sd2.0" "sdxl")

for g in "${generators[@]}"; do
    python genai.py --model_path ${g}
done