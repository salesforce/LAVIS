#!/bin/bash
TAG="gcr.io/salesforce-research-internal/lavis_streamlit_gpu"
gcloud builds submit . -t=$TAG --machine-type=n1-highcpu-32 --timeout=9000                                                                                                                  
