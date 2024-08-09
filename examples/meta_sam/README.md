# Meta SAM Example

## Acquire the model checkpoint files

The model checkpoints are not included in this build process due to their size. They should downloaded and added to the image or mounted by a shared volume.

The smallest `vit_b` model is available from (here)[https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pt].

## Building the worker image

`docker compose` should be used to build the image to ensure the root project context is added to the build process.

```bash
cd examples/meta_sam/
docker compose build
```

## Running a worker

Check the model options

```bash
docker run -it --rm label-studio-berq-sam-ml-backend --help
```

Run a model on `sam` queue. 

> This requires an active redis server.

```bash
docker run -it --rm -v ~/projects/models/sam:/models --network=host label-studio-berq-sam-ml-backend --model-path=/models/sam_vit_b_01ec64.pt --model-type=vit_b sam
```