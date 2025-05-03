# 定义批次数组
batches=(
  "0 100"
  "100 200"
  "200 300"
  "300 400"
  "400 500"
  "500 561"
)

# 循环启动最多6个并行任务
for batch in "${batches[@]}"; do
    read start end <<< "$batch"
    echo "▶️ Running batch: $start to $end"
    python scripts/download_aia_images.py --start "$start" --end "$end" &
done

# 等待所有后台任务完成
wait
echo "✅ All batches finished."