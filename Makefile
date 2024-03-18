USERNAME = $(USER)
USER_ID = $(shell id -u)
USER_GID = $(shell id -g)

%:
	@DOCKER_BUILDKIT=1 docker build -f env/Dockerfile --build-arg 'TARGET=$*' --build-arg 'USERNAME=$(USERNAME)' --build-arg 'USER_ID=${USER_ID}' --build-arg 'USER_GID=${USER_GID}' -t rnn-$* .

update-miniconda-installer:
	@ curl -o env/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
	@ chmod a+x env/miniconda.sh
