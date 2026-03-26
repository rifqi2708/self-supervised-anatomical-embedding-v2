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

git config --global user.email "rifqiab2708@gmail.com"
git config --global user.name "rifqi2708"

#input masks
gdown --id 1fBLv7I8jg8yvAnkdKREBbESV6Vrx9833 -O masks.zip && \
unzip masks.zip -d data/raw_data/NIH_lymph_node -x "__MACOSX/*" && \
rm masks.zip

#quadra dataset
gdown --id 1Vvuq7ni_Qs3JNzgqgaYFYlkpyEApugGk -O quadra_dataset.zip && \
unzip quadra_dataset.zip && \
rm quadra_dataset.zip && \
mv quadra_dataset data
