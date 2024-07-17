#!/bin/bash

# ユーザー情報の取得
nb_user=${USER}
nb_uid=$(id -u)
nb_gid=$(id -g)

# イメージ名とコンテナ名の設定
image_name=${USER}/lavis
container_name=lavis_${nb_user}

# Dockerコンテナの起動（バックグラウンド実行）
docker run \
    -it \
    --rm \
    --runtime=nvidia \
    --ipc=host \
    --net=host \
    -v ./:/app \
    -e HOME=/app \
    --user ${nb_uid}:${nb_gid} \
    --name=${container_name} \
    ${image_name} bash -c "jupyter lab --ip=0.0.0.0 --allow-root --no-browser"
