FROM python:3

WORKDIR /usr/src/app

# copy and install label-studio-be
COPY --from=project . .
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install .[full]

# set workdir to to the model and install model requirements
WORKDIR /usr/src/app/examples/meta_sam
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt

# worker launch point
# docker run <image-name> --help
# docker run <image-name> --model-path=<model_chkpoint> --model-type=<vit_b> sam
ENTRYPOINT [ "python3", "worker.py"]
