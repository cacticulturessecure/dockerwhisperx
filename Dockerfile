# syntax=docker/dockerfile:1
ARG WHISPER_MODEL=base
ARG LANG=en
ARG UID=1001
ARG VERSION=EDGE
ARG RELEASE=0

# These ARGs are for caching stage builds in CI
ARG LOAD_WHISPER_STAGE=load_whisper
ARG NO_MODEL_STAGE=no_model

# Cache home settings
ARG CACHE_HOME=/.cache
ARG CONFIG_HOME=/.config
ARG TORCH_HOME=${CACHE_HOME}/torch
ARG HF_HOME=${CACHE_HOME}/huggingface

########################################
# Base stage
########################################
FROM python:3.11-slim AS base

ARG TARGETARCH
ARG TARGETVARIANT
ARG TARGETPLATFORM

# Install CUDA and dependencies
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg2 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-toolkit-12-0 libcudnn8 \
    $([ "$TARGETPLATFORM" = "linux/arm64" ] && echo "libgomp1=12.2.0-14 libsndfile1=1.2.0-1") && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f cuda-keyring_1.1-1_all.deb

########################################
# Build stage
########################################
FROM base AS build

ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /app

# Copy required files for installation
COPY requirements.txt /app/
COPY setup.py /app/
COPY MANIFEST.in /app/
COPY whisperX /app/whisperX/

# Install under /root/.local
ARG PIP_USER="true"
ARG PIP_NO_WARN_SCRIPT_LOCATION=0
ARG PIP_ROOT_USER_ACTION="ignore"
ARG PIP_NO_COMPILE="true"
ARG PIP_DISABLE_PIP_VERSION_CHECK="true"

# Install requirements
RUN --mount=type=cache,id=pip-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/pip \
    pip install -U --force-reinstall pip setuptools wheel && \
    pip install -U --extra-index-url https://download.pytorch.org/whl/cu120 \
    torch==2.1.2 torchaudio==2.1.2 \
    pyannote.audio==3.1.1 \
    "numpy<2.0"

# Install requirements.txt
RUN --mount=type=cache,id=pip-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/pip \
    pip install -r requirements.txt

# Install whisperX with explicit PYTHONPATH
RUN --mount=type=cache,id=pip-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/pip \
    PYTHONPATH=/app pip install -e . && \
    find "/root/.local" -name '*.pyc' -print0 | xargs -0 rm -f || true ; \
    find "/root/.local" -type d -name '__pycache__' -print0 | xargs -0 rm -rf || true ;

########################################
# Final stage for no_model
########################################
FROM base AS no_model

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# We don't need them anymore
RUN pip3.11 uninstall -y pip wheel && \
    rm -rf /root/.cache/pip

# Create user
ARG UID
RUN groupadd -g $UID $UID && \
    useradd -l -u $UID -g $UID -m -s /bin/sh -N $UID

ARG CACHE_HOME
ARG CONFIG_HOME
ARG TORCH_HOME
ARG HF_HOME
ENV XDG_CACHE_HOME=${CACHE_HOME}
ENV TORCH_HOME=${TORCH_HOME}
ENV HF_HOME=${HF_HOME}

RUN install -d -m 775 -o $UID -g 0 /licenses && \
    install -d -m 775 -o $UID -g 0 ${CACHE_HOME} && \
    install -d -m 775 -o $UID -g 0 ${CONFIG_HOME}

# Copy ffmpeg and dumb-init
COPY --link --from=ghcr.io/jim60105/static-ffmpeg-upx:7.0-1 /ffmpeg /usr/local/bin/
COPY --link --from=ghcr.io/jim60105/static-ffmpeg-upx:7.0-1 /dumb-init /usr/local/bin/

# Copy licenses with corrected paths
COPY --link --chown=$UID:0 --chmod=775 LICENSE /licenses/LICENSE
COPY --link --chown=$UID:0 --chmod=775 whisperX.LICENSE /licenses/whisperX.LICENSE

# Copy dependencies and code
COPY --link --chown=$UID:0 --chmod=775 --from=build /root/.local /home/$UID/.local

ENV PATH="/home/$UID/.local/bin:$PATH"
ENV PYTHONPATH="/home/$UID/.local/lib/python3.11/site-packages"

ARG WHISPER_MODEL
ENV WHISPER_MODEL=
ARG LANG
ENV LANG=

WORKDIR /app

VOLUME [ "/app" ]

USER $UID

STOPSIGNAL SIGINT

ENTRYPOINT [ "dumb-init", "--", "/bin/sh", "-c", "whisperx \"$@\"" ]

########################################
# load_whisper stage
########################################
FROM ${NO_MODEL_STAGE} AS load_whisper

ARG TORCH_HOME
ARG HF_HOME

# Preload vad model
RUN python3 -c 'from whisperx.vad import load_vad_model; load_vad_model("cpu");'

# Preload fast-whisper
ARG WHISPER_MODEL
RUN python3 -c 'import faster_whisper; model = faster_whisper.WhisperModel("'${WHISPER_MODEL}'")'

########################################
# load_align stage
########################################
FROM ${LOAD_WHISPER_STAGE} AS load_align

ARG TORCH_HOME
ARG HF_HOME

# Copy and run alignment model loader
COPY --chown=$UID:0 --chmod=775 load_align_model.py /app/load_align_model.py
ARG LANG
RUN for i in ${LANG}; do echo "Aligning lang $i"; python3 /app/load_align_model.py "$i"; done

########################################
# Final stage with model
########################################
FROM ${NO_MODEL_STAGE} AS final

ARG UID
ARG CACHE_HOME

COPY --link --chown=$UID:0 --chmod=775 --from=load_align ${CACHE_HOME} ${CACHE_HOME}

ARG WHISPER_MODEL
ENV WHISPER_MODEL=${WHISPER_MODEL}
ARG LANG
ENV LANG=${LANG}

ENTRYPOINT [ "dumb-init", "--", "/bin/sh", "-c", "LANG=$(echo ${LANG} | cut -d ' ' -f1); whisperx --model \"${WHISPER_MODEL}\" --language \"${LANG}\" \"$@\"" ]

ARG VERSION
ARG RELEASE
LABEL name="whisperX" \
    vendor="Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew" \
    version=${VERSION} \
    release=${RELEASE} \
    summary="WhisperX: Time-Accurate Speech Transcription of Long-Form Audio" \
    description="Automatic Speech Recognition with Word-Level Timestamps (and Speaker Diarization)"
