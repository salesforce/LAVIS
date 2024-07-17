FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /root
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Jupyter Labのインストール（もしrequirements.txtに含まれていない場合）
RUN pip3 install jupyterlab

# Jupyter Labの設定ディレクトリを作成
RUN mkdir -p /root/.jupyter

WORKDIR /app
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
