FROM tensorflow/tensorflow:2.4.0-gpu-jupyter
MAINTAINER Louis Tiao <louistiao@gmail.com>

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get update && apt-get install -y --no-install-recommends \
        msttcorefonts \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-xetex \
        cm-super \
        dvipng \
        pandoc \
        imagemagick \
        ffmpeg \
        graphviz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ~/.cache/matplotlib

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==21.0.1 && \
    python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
