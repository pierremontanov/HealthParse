#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# DocIQ – Single-Host Deployment Script
# ══════════════════════════════════════════════════════════════════
# Usage:
#   ./deploy/deploy.sh              # Build + start production stack
#   ./deploy/deploy.sh --build      # Force rebuild images
#   ./deploy/deploy.sh --down       # Stop and remove containers
#   ./deploy/deploy.sh --status     # Show running containers + health
#   ./deploy/deploy.sh --logs       # Tail combined logs
# ══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILES="-f $PROJECT_ROOT/docker-compose.yml -f $PROJECT_ROOT/docker-compose.prod.yml"

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Pre-flight checks ───────────────────────────────────────────

preflight() {
    local ok=true

    if ! command -v docker &>/dev/null; then
        error "Docker is not installed. See https://docs.docker.com/engine/install/"
        ok=false
    fi

    if ! docker compose version &>/dev/null; then
        error "Docker Compose v2 plugin not found."
        ok=false
    fi

    if ! docker info &>/dev/null 2>&1; then
        error "Docker daemon is not running."
        ok=false
    fi

    $ok || exit 1
}

# ── .env setup ───────────────────────────────────────────────────

ensure_env() {
    local env_file="$PROJECT_ROOT/.env"
    if [ ! -f "$env_file" ]; then
        warn ".env not found – copying from deploy/.env.production"
        cp "$SCRIPT_DIR/.env.production" "$env_file"
        info "Created $env_file – review and adjust before production use."
    fi
}

# ── Output directory ─────────────────────────────────────────────

ensure_dirs() {
    mkdir -p "$PROJECT_ROOT/output"
    mkdir -p "$PROJECT_ROOT/data"
    info "Output and data directories ready."
}

# ── Commands ─────────────────────────────────────────────────────

cmd_up() {
    local build_flag=""
    if [[ "${1:-}" == "--build" ]]; then
        build_flag="--build"
    fi

    info "Starting DocIQ production stack..."
    docker compose $COMPOSE_FILES up -d $build_flag

    info "Waiting for health check..."
    local retries=0
    local max_retries=30
    until docker compose $COMPOSE_FILES exec -T api curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        retries=$((retries + 1))
        if [ $retries -ge $max_retries ]; then
            error "API did not become healthy within ${max_retries}s"
            docker compose $COMPOSE_FILES logs --tail=20 api
            exit 1
        fi
        sleep 1
    done

    info "DocIQ is running and healthy."
    echo ""
    cmd_status
}

cmd_down() {
    info "Stopping DocIQ stack..."
    docker compose $COMPOSE_FILES down
    info "Stopped."
}

cmd_status() {
    echo "─── Container status ───────────────────────────────────"
    docker compose $COMPOSE_FILES ps
    echo ""
    echo "─── Health endpoint ────────────────────────────────────"
    curl -sf http://localhost:${NGINX_PORT:-80}/health 2>/dev/null \
        | python3 -m json.tool 2>/dev/null \
        || warn "Could not reach /health (is the stack running?)"
}

cmd_logs() {
    docker compose $COMPOSE_FILES logs -f --tail=100
}

# ── Main ─────────────────────────────────────────────────────────

main() {
    cd "$PROJECT_ROOT"
    preflight

    case "${1:-}" in
        --down)
            cmd_down
            ;;
        --status)
            cmd_status
            ;;
        --logs)
            cmd_logs
            ;;
        --build)
            ensure_env
            ensure_dirs
            cmd_up --build
            ;;
        --help|-h)
            echo "Usage: $0 [--build|--down|--status|--logs|--help]"
            echo ""
            echo "  (default)   Build (if needed) and start the production stack"
            echo "  --build     Force rebuild images before starting"
            echo "  --down      Stop and remove all containers"
            echo "  --status    Show container status and health"
            echo "  --logs      Tail combined container logs"
            ;;
        *)
            ensure_env
            ensure_dirs
            cmd_up
            ;;
    esac
}

main "$@"
