#!/bin/bash

# ユーザー情報の取得
nb_user=${USER}
nb_uid=$(id -u)
nb_gid=$(id -g)

# イメージ名とコンテナ名の設定
image_name=${USER}/lavis

docker build --build-arg JUPYTER_PASSWORD=password -t $image_name .
