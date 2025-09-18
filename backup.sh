#!/bin/bash

# =========================
# 配置项（请改成你的信息）
# =========================
GIT_USER="Mr8wx"       # GitHub 用户名
GIT_EMAIL="wxing.inef@gmail.com"    # GitHub 绑定邮箱
REPO_URL="https://github.com/Mr81wx/QTNBA.git"  # 仓库地址（SSH 或 HTTPS）
BRANCH="main"                         # 目标分支
# =========================

# 设置 Git 用户信息
git config --global user.name "$GIT_USER"
git config --global user.email "$GIT_EMAIL"

# 初始化仓库（如果还没有）
if [ ! -d ".git" ]; then
  git init
  git branch -M $BRANCH
fi

# 添加远程仓库（如果还没设置）
if ! git remote | grep -q origin; then
  git remote add origin "$REPO_URL"
fi

# 添加所有文件并提交
git add .
git commit -m "Auto backup on $(date '+%Y-%m-%d %H:%M:%S')" || echo "⚠️ 没有新变化可提交"

# 推送到 GitHub
git push -u origin $BRANCH
