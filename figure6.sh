PYTHON_PATH="/opt/anaconda3/bin/python"  ### your python path
SCRIPT_PATH="/CausalGraphCut/sim_MSE.py"  ### the path of sim_MSE.py

cor_types=("example1" "example2" "example3")
pattern_types=("circle_hexagon" "hexagon" "fan_hexagon" "rectangle_hexagon")
for pattern in "${pattern_types[@]}"; do
    case $pattern in
        "hexagon")
            grid=12
            ;;
        "fan_hexagon")
            grid=19
            ;;
        "circle_hexagon")
            grid=10
            ;;
        "rectangle_hexagon")
            grid=30
            ;;
        # Add more patterns with corresponding grid sizes here
        *)
            echo "Unknown pattern: $pattern"
            continue
            ;;
    esac
    for cor_type in "${cor_types[@]}"; do
        for rho in $(seq 0.9 -0.1 0.7); do
            echo "Running simulation with pattern=$pattern (size=$grid), cor-type=$cor_type and rho=$rho"
            $PYTHON_PATH $SCRIPT_PATH --pattern="$pattern" --rho=$rho --cor-type="$cor_type" --sample-num=400 --grid-size=$grid
        done
    done
done

