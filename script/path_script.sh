find app/common app/controller app/models app/repository app/schemas app/services app/main.py app/api app/core \
  -type f \
  -not -path "app/services/unilm/*" \
  -not -path "*/__pycache__/*" \
  -not -name "*.pyc" \
  -not -name "*.pyo" \
| while read f; do
    echo "===== $f ====="
    cat "$f"
    echo -e "\n"
done > output.txt
