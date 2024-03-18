USERNAME = $(USER)
USER_ID = $(shell id -u)
USER_GID = $(shell id -g)

%:
	@DOCKER_BUILDKIT=1 docker build -f env/Dockerfile --build-arg 'TARGET=$*' --build-arg 'USERNAME=$(USERNAME)' --build-arg 'USER_ID=${USER_ID}' --build-arg 'USER_GID=${USER_GID}' -t rnn-$* .
