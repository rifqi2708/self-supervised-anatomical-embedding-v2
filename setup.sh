pip install gdown && \
    gdown 1LH9E5D273kOJXrUmBv_s2hXuOZV-dR65 -O weights.zip && \
    unzip weights.zip && \
    mv Self-supervised_Anatomical_Embeddings/checkpoints . && \
    mv Self-supervised_Anatomical_Embeddings/data . && \
    rm -r Self-supervised_Anatomical_Embeddings weights.zip 

pip install ipython

# git branch dev origin/dev
# git checkout dev
ipython misc/lymphnode_preprocess_crop_multi_process.py

git config --global user.email "rifqiab2708@gmail.com"
git config --global user.name "rifqi2708"


# #quadra dataset males
# gdown 1R51bptSQLkhziDPzAAagv0KvVGGspcSF -O quadra_dataset_males.zip && \
# unzip quadra_dataset_males.zip && \
# rm quadra_dataset_males.zip && \
# mv quadra_dataset_males data

# #quadra dataset females
# gdown 1hR-Df0pt9FUwd_i4IOsyFYwohsehnJyX -O quadra_dataset_females.zip && \
# unzip quadra_dataset_females.zip && \
# rm quadra_dataset_females.zip && \
# mv quadra_dataset_females data

#quadra dataset females
gdown 1ZxCDER7jAdn5fgMvexGTiRH7xz9RJM6- -O quadra_dataset_cropped.zip && \
unzip quadra_dataset_cropped.zip && \
rm quadra_dataset_cropped.zip && \
mv quadra_dataset_cropped data
