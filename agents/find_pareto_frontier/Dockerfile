# Build with run-agent.sh --build-only: it supplies a minimal, credential-free context.
ARG PYTHON_IMAGE=python:3.14.5-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.19
ARG NODE_IMAGE=node:26.7.0-bookworm-slim
FROM ${UV_IMAGE} AS uv
FROM ${NODE_IMAGE} AS agent
ARG CODEX_VERSION=0.166.0
# Install the native package as required: npm may silently drop failed optional downloads.
RUN set -eu; \
    npm_install() { npm install --global --omit=optional --no-audit --no-fund \
        --fetch-retries=3 --fetch-retry-mintimeout=2000 --fetch-retry-maxtimeout=10000 "$@"; }; \
    npm_install "@openai/codex@${CODEX_VERSION}"; \
    native_package="$(node -e ' \
        const {arch, platform} = process; \
        if (platform !== "linux" || !["x64", "arm64"].includes(arch)) \
            throw new Error("Unsupported Codex build platform: " + platform + "/" + arch); \
        const name = "@openai/codex-linux-" + arch; \
        const manifest = require("/usr/local/lib/node_modules/@openai/codex/package.json"); \
        const spec = manifest.optionalDependencies?.[name]; \
        if (typeof spec !== "string" || !spec) throw new Error("Missing native package declaration: " + name); \
        process.stdout.write(name + "@" + spec); \
    ')"; \
    npm_install "$native_package"; \
    codex --version; \
    test "$(codex --version)" = "codex-cli ${CODEX_VERSION}"

FROM ${PYTHON_IMAGE}
ARG RUST_VERSION=1.96.0
ARG CODEX_VERSION=0.166.0

# HTTPS avoids proxy handling of plain HTTP repositories. Disable pipelining
# and retry transient downloads without relaxing repository or TLS verification.
RUN set -eu; \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
        /etc/apt/sources.list.d/debian.sources; \
    apt_get() { apt-get -o Acquire::Retries=3 \
        -o Acquire::http::Pipeline-Depth=0 \
        -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 "$@"; }; \
    apt_get update --error-on=any; \
    apt_get install --yes --no-install-recommends \
        bash build-essential ca-certificates curl git jq procps ripgrep \
        time util-linux; \
    rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=agent /usr/local/bin/node /usr/local/bin/node
COPY --from=agent /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo-seed
ENV PATH=/opt/venv/bin:/opt/cargo-seed/bin:/usr/local/bin:/usr/bin:/bin
RUN curl --fail --silent --show-error --location https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain "${RUST_VERSION}" --no-modify-path \
    && rm /tmp/rustup.sh

# The project's optional vllm group is deliberately excluded.
COPY pyproject.toml uv.lock /opt/python-project/
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --project /opt/python-project \
        --locked --no-default-groups --group dev --no-group vllm \
        --python /usr/local/bin/python --no-install-project \
    && python -c 'import sys, importlib.util; assert sys.version_info[:2] == (3,14); assert importlib.util.find_spec("vllm") is None; import matplotlib, networkx'

# Cache exactly the workspace dependencies; application source is supplied per run.
COPY Cargo.toml Cargo.lock /opt/cargo-project/
COPY crates/simthesizer-core/Cargo.toml /opt/cargo-project/crates/simthesizer-core/
COPY crates/simthesizer-cli/Cargo.toml /opt/cargo-project/crates/simthesizer-cli/
RUN mkdir -p /opt/cargo-project/crates/simthesizer-core/src \
        /opt/cargo-project/crates/simthesizer-cli/src \
    && touch /opt/cargo-project/crates/simthesizer-core/src/lib.rs \
    && printf 'fn main() {}\n' > /opt/cargo-project/crates/simthesizer-cli/src/main.rs \
    && cargo fetch --locked --manifest-path /opt/cargo-project/Cargo.toml \
    && chmod -R a+rX /opt/cargo-seed /opt/rustup /opt/venv

COPY entrypoint.sh launcher.py baseline_store.py /opt/sim-agent/
RUN chmod 0755 /opt/sim-agent/entrypoint.sh \
    && mkdir -p /workspace/simthesizer /workspace/task /workspace/inputs \
        /workspace/artifacts /workspace/cache /home/agent \
    && printf 'export PATH=%s\n' "$PATH" > /etc/profile.d/simthesizer.sh \
    && /opt/sim-agent/entrypoint.sh --check-env \
    && codex exec --help > /tmp/codex-help \
    && for flag in --ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check \
        --dangerously-bypass-approvals-and-sandbox --json --output-last-message; do \
        rg --quiet -- "$flag" /tmp/codex-help || exit 1; \
    done \
    && rm /tmp/codex-help

ENV HOME=/home/agent CODEX_HOME=/home/agent/.codex \
    CARGO_HOME=/workspace/cache/cargo CARGO_TARGET_DIR=/workspace/cache/target \
    CARGO_NET_OFFLINE=true UV_PROJECT_ENVIRONMENT=/opt/venv UV_NO_SYNC=1 \
    UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never UV_CACHE_DIR=/workspace/cache/uv \
    PYTHONPATH=/workspace/simthesizer/scripts PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/workspace/cache/matplotlib \
    XDG_CONFIG_HOME=/home/agent/.config XDG_DATA_HOME=/home/agent/.local/share \
    XDG_STATE_HOME=/home/agent/.local/state XDG_CACHE_HOME=/workspace/cache/xdg
WORKDIR /workspace/simthesizer
ENTRYPOINT ["/opt/sim-agent/entrypoint.sh"]

# Identity changes must not invalidate unrelated dependency installation layers.
ARG BUILD_INPUT_SHA256
LABEL io.simthesizer.agent.build-input-sha256=${BUILD_INPUT_SHA256} \
      io.simthesizer.agent.codex-version=${CODEX_VERSION}
