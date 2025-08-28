find app/common app/controller app/models app/repository app/schemas app/services -type f \
  -not -path "app/services/unilm/*" | while read f; do
    echo "===== $f ====="
    cat "$f"
    echo -e "\n"
done > output.txt