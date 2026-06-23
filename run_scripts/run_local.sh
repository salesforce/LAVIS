#!/bin/bash
conda create -n lavis_local python=3.8                                                                                                                                      
conda init bash
# YOU MAY NEED TO RESTART THE SHELL FOR CONDA INIT TO TAKE EFFECT
echo "Restart shell and continue the rest"
exit 1
conda activate lavis_local
git clone https://github.com/MetaMind/LAVIS.git
cd LAVIS
git clone https://github.com/CompVis/stable-diffusion.git                                                                                                                    
git checkout lavis_diffusion
pip install -r requirements-dev.txt
export PYTHONPATH=./:$PYTHONPATH:./stable-diffusion
streamlit run --server.port 8080 app/main.py
