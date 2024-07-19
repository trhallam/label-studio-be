#!/bin/bash

# Build and run tests for all images locally
# ./make.sh

DOCKER_BASE=app
GIT_REV=$(git rev-parse --short HEAD)
ARG_LEN=${#@}

echo "Building: ${DOCKER_BASE}"
if [[ -d $DOCKER_BASE ]]; then
    dir=${DOCKER_BASE}
    echo "Building: $dir, $GIT_REV"
    # lint the dockerfile
    docker run --rm -i hadolint/hadolint < ${dir}/Dockerfile
    # build with local git commit revision
    docker buildx build $dir --pull -t trhallam/label-studio-be:${GIT_REV}
    # run unittests
    # docker run \
        # --env-file ${dir}/test_config.env \
        # -v $DOCKER_BASE:/tmp/ \
        # epcc/ansibleubuntu${dir}:${GIT_REV} \
        # python3 -m unittest discover --verbose /tmp/test
fi
