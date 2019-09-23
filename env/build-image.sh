#!/bin/sh

# Create docker image containing qserv deployment tools 

# @author  Benjamin Roziere 
# @author  Fabrice Jammes

set -e
#set -x

DIR=$(cd "$(dirname "$0")"; pwd -P)

GIT_HASH="$(git describe --dirty --always)"
TAG=${cpu-${GIT_HASH}}

IMAGE="supernnova/supernnova:$TAG"

echo "Building image $IMAGE"

docker build --tag "$IMAGE" -f Dockerfile.cpu .
docker push "$IMAGE"
echo "$IMAGE pushed to Docker Hub"
