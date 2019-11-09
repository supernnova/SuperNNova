#!/bin/bash
set -e


if [ -z "${USER}" ]; then
  echo "We need USER to be set!"; exit 100
fi

# if both not set we do not need to do anything
if [ -z "${HOST_USER_ID}" -a -z "${HOST_USER_GID}" ]; then
    echo "Nothing to do here." ; exit 0
fi

# reset user_?id to either new id or if empty old (still one of above
# might not be set)
USER_ID=${HOST_USER_ID:=$USER_ID}
USER_GID=${HOST_USER_GID:=$USER_GID}

LINE=$(grep -F "${USER}" /etc/passwd)
# replace all ':' with a space and create array
array=( ${LINE//:/ } )

# home is 5th element
USER_HOME=/u/home

sed -i -e "s/^${USER}:\([^:]*\):[0-9]*:[0-9]*/${USER}:\1:${USER_ID}:${USER_GID}/"  /etc/passwd
sed -i -e "s/^${USER}:\([^:]*\):[0-9]*/${USER}:\1:${USER_GID}/"  /etc/group

chown $HOST_USER_ID:$HOST_USER_GID $USER_HOME
chown $HOST_USER_ID:$HOST_USER_GID /u/home/.zshrc
chown $HOST_USER_ID:$HOST_USER_GID /u/home/.oh-my-zsh

_term() { 
  echo "Caught SIGTERM signal!" 
  kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM
# PATH is reset on user switch but we need to preserve


su -p "${USER}"

