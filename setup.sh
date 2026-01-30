pip install gdown && \
    gdown 1LH9E5D273kOJXrUmBv_s2hXuOZV-dR65 -O weights.zip && \
    unzip weights.zip && \
    mv Self-supervised_Anatomical_Embeddings/checkpoints . && \
    mv Self-supervised_Anatomical_Embeddings/data . && \
    rm -r Self-supervised_Anatomical_Embeddings weights.zip 

pip install ipython

git branch dev origin/dev
git checkout dev
ipython misc/lymphnode_preprocess_crop_multi_process.py

git config --global user.email "sunyu0410@gmail.com"
git config --global user.name "Yu Sun"
