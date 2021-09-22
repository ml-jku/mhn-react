# Docker Image

Build docker image using following command.
[`DOCKER_BUILDKIT`](https://docs.docker.com/develop/develop-images/build_enhancements/) results in faster builds but is Linux only.

```
DOCKER_BUILDKIT=1 docker build -t mhnreact:latest -f Dockerfile ../..
```
