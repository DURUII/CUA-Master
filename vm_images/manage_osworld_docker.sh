#!/usr/bin/env bash
set -euo pipefail

STORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATALOG="$STORE_DIR/catalog.json"
DOCKER_IMAGE="${OSWORLD_DOCKER_IMAGE:-happysixd/osworld-docker}"

usage() {
  cat <<'EOF'
Usage:
  manage_osworld_docker.sh list
  manage_osworld_docker.sh info IMAGE_ID
  manage_osworld_docker.sh start IMAGE_ID
  manage_osworld_docker.sh stop IMAGE_ID
  manage_osworld_docker.sh status IMAGE_ID
  manage_osworld_docker.sh logs IMAGE_ID

Optional environment variables:
  OSWORLD_VNC_PORT=8006
  OSWORLD_SERVER_PORT=5000
  OSWORLD_CHROMIUM_PORT=9222
  OSWORLD_VLC_PORT=8080
  OSWORLD_BIND_ADDRESS=127.0.0.1
  OSWORLD_RAM_SIZE=8G
  OSWORLD_CPU_CORES=4
  OSWORLD_DISK_SIZE=64G
EOF
}

container_name() {
  printf 'osworld-%s\n' "$1" | tr '[:upper:]_.' '[:lower:]---'
}

image_record() {
  local image_id="$1"
  jq -er --arg image_id "$image_id" '.images[$image_id]' "$CATALOG"
}

case "${1:-}" in
  list)
    jq -r '.images | to_entries[] | [.key, .value.os.type, .value.os.version, .value.path] | @tsv' "$CATALOG"
    ;;
  info)
    test "$#" -eq 2 || { usage; exit 2; }
    image_record "$2"
    ;;
  start)
    test "$#" -eq 2 || { usage; exit 2; }
    record="$(image_record "$2")"
    os_type="$(jq -r '.os.type' <<<"$record")"
    if [[ "$os_type" != "Windows" && "$os_type" != "Ubuntu" ]]; then
      echo "OSWorld Docker provider supports Ubuntu/Windows only; $2 is $os_type." >&2
      echo "macOS qcow2 requires a compliant Apple host and a dedicated provider." >&2
      exit 2
    fi
    relative_path="$(jq -r '.path' <<<"$record")"
    disk="$STORE_DIR/$relative_path"
    test -f "$disk" || { echo "Missing disk: $disk" >&2; exit 1; }
    name="$(container_name "$2")"
    if docker container inspect "$name" >/dev/null 2>&1; then
      echo "Container already exists: $name" >&2
      exit 1
    fi

    docker_args=(
      run -d --name "$name" --privileged
      -e "DISK_SIZE=${OSWORLD_DISK_SIZE:-64G}"
      -e "RAM_SIZE=${OSWORLD_RAM_SIZE:-8G}"
      -e "CPU_CORES=${OSWORLD_CPU_CORES:-4}"
      -v "$disk:/System.qcow2:ro"
      -p "${OSWORLD_BIND_ADDRESS:-127.0.0.1}:${OSWORLD_VNC_PORT:-8006}:8006"
      -p "${OSWORLD_BIND_ADDRESS:-127.0.0.1}:${OSWORLD_SERVER_PORT:-5000}:5000"
      -p "${OSWORLD_BIND_ADDRESS:-127.0.0.1}:${OSWORLD_CHROMIUM_PORT:-9222}:9222"
      -p "${OSWORLD_BIND_ADDRESS:-127.0.0.1}:${OSWORLD_VLC_PORT:-8080}:8080"
    )
    for proxy_var in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
      if [[ -n "${!proxy_var:-}" ]]; then
        docker_args+=(-e "$proxy_var")
      fi
    done
    if [[ -e /dev/net/tun ]]; then
      docker_args+=(--device /dev/net/tun)
    fi
    if [[ -e /dev/kvm ]]; then
      docker_args+=(--device /dev/kvm)
    else
      docker_args+=(-e KVM=N)
      echo "Warning: /dev/kvm is unavailable; software emulation will be slow." >&2
    fi
    docker_args+=("$DOCKER_IMAGE")
    docker "${docker_args[@]}"
    echo "Container: $name"
    echo "noVNC:     http://${OSWORLD_BIND_ADDRESS:-127.0.0.1}:${OSWORLD_VNC_PORT:-8006}"
    echo "VM API:    http://${OSWORLD_BIND_ADDRESS:-127.0.0.1}:${OSWORLD_SERVER_PORT:-5000}"
    ;;
  stop)
    test "$#" -eq 2 || { usage; exit 2; }
    docker rm -f "$(container_name "$2")"
    ;;
  status)
    test "$#" -eq 2 || { usage; exit 2; }
    docker container inspect --format '{{.State.Status}}' "$(container_name "$2")"
    ;;
  logs)
    test "$#" -eq 2 || { usage; exit 2; }
    docker logs -f "$(container_name "$2")"
    ;;
  *)
    usage
    exit 2
    ;;
esac
