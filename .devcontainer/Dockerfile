FROM python:3.10-bookworm

ARG USERNAME=remote
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    && apt-get update -y \
    # Install necessary packages
    && apt-get install -y \
    sudo \
    curl \
    build-essential cmake ncurses-dev libncurses5-dev libpcre2-dev gettext \
    python3-dev python3-pythran gfortran libopenblas-dev patchelf \
    #
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the default user
USER $USERNAME
