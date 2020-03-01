FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN pip install --upgrade pip && \
    pip install dill tqdm requests # matplotlib scipy tensorflow-gpu==1.12.0

USER $NB_UID

CMD bash