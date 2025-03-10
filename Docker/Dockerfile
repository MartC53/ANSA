# Use the base PyTorch Jupyter image
FROM quay.io/jupyter/pytorch-notebook:cuda12-ubuntu-24.04

# Set working directory inside the container
WORKDIR /home/jovyan/work

# Copy notebooks into the container
COPY ./*.ipynb /home/jovyan/work/

# Make directory for mounting datasets
RUN mkdir -p /home/jovyan/work/datasets

# Expose Jupyter Notebook's default port
EXPOSE 8888

# Run Jupyter Notebook on container startup
CMD ["start-notebook.sh", "--NotebookApp.token=''"]
