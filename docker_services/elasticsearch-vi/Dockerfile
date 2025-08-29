# ---------- Global ARGs ----------
ARG ES_VERSION=8.7.0

# ===== Builder: Debian Buster (GCC 8) so GLIBCXX_3.4.26 =====
FROM debian:buster AS builder
ARG ES_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Use archived repos (buster is EOL) and disable expiry checks
RUN printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99no-check-valid && \
    sed -i 's|deb.debian.org/debian|archive.debian.org/debian|g' /etc/apt/sources.list || true && \
    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list || true && \
    sed -i '/buster-updates/d' /etc/apt/sources.list || true && \
    apt-get -o Acquire::Check-Valid-Until=false update -y && \
    apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates pkg-config \
      openjdk-11-jdk maven && \
    rm -rf /var/lib/apt/lists/*

# Build coccoc-tokenizer (JNI)
WORKDIR /tmp
RUN git clone https://github.com/coccoc/coccoc-tokenizer.git
WORKDIR /tmp/coccoc-tokenizer/build
RUN cmake -DBUILD_JAVA=1 .. && make -j"$(nproc)" && make install

# Sanity check: what did we install?
RUN ls -al /usr/local/lib && ls -al /usr/local/share/tokenizer/dicts

# Fetch plugin zip for ES_VERSION
RUN wget -O /tmp/elasticsearch-analysis-vietnamese.zip \
  "https://github.com/duydo/elasticsearch-analysis-vietnamese/releases/download/v${ES_VERSION}/elasticsearch-analysis-vietnamese-${ES_VERSION}.zip"

# ===== Final: ES with plugin & JNI =====
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION}
USER root

ARG COCCOC_INSTALL_PATH=/usr/local
ARG COCCOC_DICT_PATH=${COCCOC_INSTALL_PATH}/share/tokenizer/dicts

# Copy JNI + dicts
COPY --from=builder /usr/local/lib/libcoccoc_tokenizer_jni.so /usr/lib/
COPY --from=builder ${COCCOC_DICT_PATH} ${COCCOC_DICT_PATH}

# Ensure ES finds the JNI
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH}"
ENV ES_JAVA_OPTS="-Xms2g -Xmx2g -Djava.library.path=/usr/lib:/usr/local/lib"

# Install plugin
COPY --from=builder /tmp/elasticsearch-analysis-vietnamese.zip /elasticsearch-analysis-vietnamese.zip
RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install --batch file:///elasticsearch-analysis-vietnamese.zip \
 && rm -f /elasticsearch-analysis-vietnamese.zip

ENV discovery.type=single-node
USER elasticsearch
