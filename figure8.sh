PYTHON_PATH="/opt/anaconda3/bin/python"
SCRIPT_PATH="/CausalGraphCut/sim_ss.py"

cor_types=("example1" "example2" "example3")
pattern_types=("hexagon")

nums=(20 40 60 80 100 120 140)
for pattern in "${pattern_types[@]}"; do
    case $pattern in
        "hexagon")
            grid=12
            ;;
        # Add more patterns with corresponding grid sizes here
        *)
            echo "Unknown pattern: $pattern"
            continue
            ;;
    esac
    for cor_type in "${cor_types[@]}"; do
        for num in "${nums[@]}"; do
            echo "Running simulation with sample-num=$num"
            $PYTHON_PATH $SCRIPT_PATH --pattern="$pattern" --rho=0.5 --cor-type="$cor_type" --sample-num=$num --grid-size=$grid
        done
    done
done
