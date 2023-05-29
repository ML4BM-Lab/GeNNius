FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime 

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git\
    nano\
    wget


RUN pip install --upgrade pip


RUN pip install matplotlib pandas networkx umap-learn seaborn
RUN pip install datashader bokeh holoviews scikit-image colorcet
RUN pip install rdkit-pypi

#RUN pip install torch_geometric
# latest version available
RUN pip install torch-geometric==2.3.0 
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

RUN mkdir /wdir
WORKDIR /wdir
