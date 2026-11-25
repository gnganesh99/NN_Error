#!/usr/bin/env bash
set -u

python_cmd="${PYTHON:-python}"
# Random IV active learning: im2spec x (ss, ms) x (distance, random).
# Flags not used here are passed to the script, e.g.:
#   ./run_models.sh --python ../../.venv/bin/python --acquisition-iterations 50
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_path="${SCRIPT_PATH:-"$script_dir/iv_acqfn_script.py"}"
continue_on_error=0
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --python" >&2
                exit 2
            fi
            python_cmd="$2"
            shift 2
            ;;
        --script-path)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --script-path" >&2
                exit 2
            fi
            script_path="$2"
            shift 2
            ;;
        --continue-on-error)
            continue_on_error=1
            shift
            ;;
        --)
            shift
            extra_args+=("$@")
            break
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "$script_path" ]]; then
    echo "Model runner script was not found: $script_path" >&2
    exit 1
fi

labels=("im2spec")
cli_models=("im2spec")
acquisition_methods=("distance" "random")
scale_types=("ss" "ms")
failures=()

for i in "${!labels[@]}"; do
    label="${labels[$i]}"
    cli_model="${cli_models[$i]}"

    for scale_type in "${scale_types[@]}"; do
        for acquisition_method in "${acquisition_methods[@]}"; do
            echo
            echo "Running model '$label' with scale type '$scale_type' and acquisition method '$acquisition_method'"

            "$python_cmd" "$script_path" \
                --model-type "$cli_model" \
                --scale-type "$scale_type" \
                --acquisition-method "$acquisition_method" \
                "${extra_args[@]}"

            exit_code=$?
            if [[ $exit_code -ne 0 ]]; then
                failure="$label / $scale_type / $acquisition_method exited with code $exit_code"
                failures+=("$failure")
                echo "Warning: $failure" >&2

                if [[ $continue_on_error -eq 0 ]]; then
                    exit "$exit_code"
                fi
            fi
        done
    done
done

if [[ ${#failures[@]} -gt 0 ]]; then
    echo
    echo "Completed with failures:"
    for failure in "${failures[@]}"; do
        echo " - $failure"
    done
    exit 1
fi

echo
echo "All model runs completed successfully."
