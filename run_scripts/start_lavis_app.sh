#!/bin/sh
nohup python /lavis_app/app/backend/multimodal_search_backend.py > app/backend/search.log &
nohup python /lavis_app/app/backend/txt2image_backend.py > app/backend/imagen.log &
nohup python /lavis_app/app/backend/caption_backend.py > app/backend/caption.log &
streamlit run --server.port 8080 /lavis_app/app/main.py 
