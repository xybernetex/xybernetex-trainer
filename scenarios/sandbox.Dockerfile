# Sandbox image for the scenario suite's dedicated OpenClaw agent.
#
# OpenClaw's own sandbox image (scripts/docker/sandbox/Dockerfile in
# openclaw/openclaw, v2026.9.6) plus the tools scenarios exercise: sqlite3
# (SQL scenarios), pip, node/npm (package scenarios), an ssh client (scp
# scenarios) and procps. Built under its own tag so OpenClaw's default image
# name stays free:
#
#   docker build -t openclaw-sandbox:xyb-scenarios -f scenarios/sandbox.Dockerfile scenarios
#   openclaw config set agents.entries.scenarios.sandbox.docker.image openclaw-sandbox:xyb-scenarios
FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git jq python3 ripgrep \
    sqlite3 python3-pip nodejs npm openssh-client procps \
  && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash sandbox \
  && git config --system user.name "Scenario Agent" \
  && git config --system user.email "agent@scenarios.invalid" \
  && git config --system init.defaultBranch main
USER sandbox
WORKDIR /home/sandbox

CMD ["sleep", "infinity"]
