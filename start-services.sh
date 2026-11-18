#!/bin/bash
# 脚本概要：启动 MySQL 8.4，安装课程电商后端与客服前端依赖，并同时启动服务。
# 使用固定课程目录；日志输出到当前终端，Ctrl+C 或任一服务退出时清理两个服务。
# MySQL 由 Homebrew 后台管理，不随脚本退出而停止。
set -euo pipefail

course_dir='/Users/zhangxu/Documents/尚硅谷AI全能开发-260525/10-大模型项目之电商小二/ai0525'
backend_dir="$course_dir/ecommerce-service-backend"
frontend_dir="$course_dir/customer-service-frontend"

for command in brew uv npm; do
    command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
for directory in "$backend_dir" "$frontend_dir"; do
    [[ -d "$directory" ]] || { echo "目录不存在：$directory" >&2; exit 1; }
done

echo '启动 MySQL 8.4 服务……'
brew services start mysql@8.4

echo '安装电商后端依赖……'
(cd "$backend_dir" && uv sync)
echo '安装客服前端依赖……'
(cd "$frontend_dir" && npm install)

# 独立进程组确保退出时也能关闭 uv/npm 启动的子进程。
set -m
service_pids=()
cleanup() {
    trap '' INT TERM
    for pid in "${service_pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

(cd "$backend_dir" && exec uv run python main.py) &
service_pids+=("$!")
(cd "$frontend_dir" && exec npm run dev) &
service_pids+=("$!")
echo '两个服务已启动，启动结果请查看下方日志；按 Ctrl+C 停止。'

while true; do
    for pid in "${service_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            status=0
            wait "$pid" || status=$?
            echo '一个服务已退出，正在停止另一个服务。'
            exit "$status"
        fi
    done
    sleep 1
done
