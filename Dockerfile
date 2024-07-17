FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /root
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Jupyter Labのインストール（もしrequirements.txtに含まれていない場合）
RUN pip3 install jupyterlab

# Jupyter Labの設定ディレクトリを作成
RUN mkdir -p /root/.jupyter

# 非対話的にJupyter Labのパスワードを設定
ARG JUPYTER_PASSWORD
RUN python3 -c "from jupyter_server.auth import passwd; print(passwd('$JUPYTER_PASSWORD'))" > /tmp/jupyter_hash.txt && \
    echo "c.ServerApp.password = '$(cat /tmp/jupyter_hash.txt)'" >> /root/.jupyter/jupyter_server_config.py && \
    rm /tmp/jupyter_hash.txt

WORKDIR /app
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
