FROM tiao/development:publication as base
MAINTAINER Louis Tiao <louistiao@gmail.com>

# TODO: copy to temporary dir rather than unknown current dir
COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir torch>=1.7 gpytorch>=1.3 botorch
