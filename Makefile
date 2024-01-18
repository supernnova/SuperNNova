%:
	DOCKER_BUILDKIT=1 docker build -f env/Dockerfile --build-arg 'TARGET=$*' -t rnn-$* .
