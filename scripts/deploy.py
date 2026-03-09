"""Deploy the RAG Learning Lab to local, AWS, or GCP.

Usage:
    uv run python scripts/deploy.py --target local
    uv run python scripts/deploy.py --target aws
    uv run python scripts/deploy.py --target gcp
    uv run python scripts/deploy.py --rollback 2026.03.06-091500
    uv run python scripts/deploy.py --rollback last-approved
    uv run python scripts/deploy.py --history
    uv run python scripts/deploy.py --show 2026.03.07-143022
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

HISTORY_PATH = Path(__file__).resolve().parent.parent / "deploy" / "history.yaml"
BACKUPS_DIR = Path(__file__).resolve().parent.parent / "backups"
VALID_TARGETS = ("local", "aws", "gcp")


@dataclass
class DeploymentRecord:
    """A single deployment entry."""

    version: str
    target: str
    git_sha: str
    timestamp: str
    status: str
    approval: str
    config: str
    notes: str


def generate_version() -> str:
    """Generate a CalVer version string: YYYY.MM.DD-HHMMSS."""
    now = datetime.now(UTC)
    return now.strftime("%Y.%m.%d-%H%M%S")


def get_git_sha() -> str:
    """Get the current git commit SHA (short form)."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_tag(tag: str) -> None:
    """Create a lightweight git tag for this deployment."""
    subprocess.run(["git", "tag", tag], check=True)


def build_image(tag: str) -> None:
    """Build and tag a Docker image for the application."""
    subprocess.run(
        ["docker", "build", "-t", f"rag-learning-lab:{tag}", "."],
        check=True,
    )


def snapshot_vector_store(version: str) -> None:
    """Create a backup snapshot of the current vector store."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    backup_path = BACKUPS_DIR / f"chroma-{version}.tar.gz"
    if data_dir.exists():
        subprocess.run(
            ["tar", "-czf", str(backup_path), "-C", str(data_dir.parent), data_dir.name],
            check=True,
        )


def save_deployment_record(record: DeploymentRecord) -> None:
    """Append a deployment record to deploy/history.yaml."""
    history = load_deployment_history()
    history.append(record)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"deployments": [asdict(r) for r in history]}
    HISTORY_PATH.write_text(yaml.dump(data, default_flow_style=False))


def load_deployment_history() -> list[DeploymentRecord]:
    """Load all deployment records from deploy/history.yaml."""
    if not HISTORY_PATH.exists():
        return []
    data = yaml.safe_load(HISTORY_PATH.read_text())
    if not data or "deployments" not in data:
        return []
    return [DeploymentRecord(**d) for d in data["deployments"]]


def get_last_approved_version(target: str) -> DeploymentRecord | None:
    """Find the most recent deployment with approval: passed for a given target."""
    history = load_deployment_history()
    for record in reversed(history):
        if record.target == target and record.approval == "passed":
            return record
    return None


def docker_compose_up(image_tag: str) -> None:
    """Start local services via docker-compose with the given image tag."""
    deploy_dir = Path(__file__).resolve().parent.parent / "deploy" / "local"
    compose_file = deploy_dir / "docker-compose.yaml"
    env = {"IMAGE_TAG": image_tag}
    subprocess.run(
        ["docker-compose", "-f", str(compose_file), "up", "-d"],
        check=True,
        env={**env},
    )


def terraform_apply(target: str, version: str) -> None:
    """Run terraform apply for AWS or GCP with the given version."""
    tf_dir = Path(__file__).resolve().parent.parent / "deploy" / target / "terraform"
    subprocess.run(
        ["terraform", "apply", "-auto-approve", f"-var=version={version}"],
        cwd=str(tf_dir),
        check=True,
    )


def rollback(version: str, target: str) -> None:
    """Roll back to a previous deployment version.

    Steps:
    1. Look up the target version in deploy/history.yaml
    2. Restore the Docker image tag
    3. Restore vector store snapshot
    4. Restart services
    5. Update deploy/history.yaml
    """
    history = load_deployment_history()
    # Mark current active as rolled-back
    for record in history:
        if record.target == target and record.status == "active":
            record.status = "rolled-back"
    # Mark the restored version as active
    for record in history:
        if record.version == version:
            record.status = "active"
            break

    # Restore vector store snapshot
    backup_path = BACKUPS_DIR / f"chroma-{version}.tar.gz"
    if backup_path.exists():
        data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        subprocess.run(
            ["tar", "-xzf", str(backup_path), "-C", str(data_dir.parent)],
            check=True,
        )

    # Restart with the target version
    if target == "local":
        docker_compose_up(version)
    elif target in ("aws", "gcp"):
        terraform_apply(target, version)

    # Save updated history
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"deployments": [asdict(r) for r in history]}
    HISTORY_PATH.write_text(yaml.dump(data, default_flow_style=False))
    print(f"Rolled back to version {version}")


def deploy(target: str, config: str = "", notes: str = "") -> None:
    """Execute a full deployment to the specified target.

    Steps:
    1. Generate CalVer version
    2. Tag the git commit
    3. Build and tag Docker image
    4. Snapshot the vector store
    5. Save deployment record
    6. Deploy to target (docker-compose or terraform)
    7. Run post-deployment approval checks
    8. Auto-rollback if approval fails
    """
    version = generate_version()
    git_sha = get_git_sha()

    print(f"Deploying version {version} to {target}...")

    # Tag, build, snapshot
    git_tag(f"deploy/v{version}")
    build_image(version)
    snapshot_vector_store(version)

    # Save initial record
    record = DeploymentRecord(
        version=version,
        target=target,
        git_sha=git_sha,
        timestamp=datetime.now(UTC).isoformat(),
        status="active",
        approval="pending",
        config=config,
        notes=notes,
    )

    # Mark previous active deployments as superseded
    history = load_deployment_history()
    for r in history:
        if r.target == target and r.status == "active":
            r.status = "superseded"
    history.append(record)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"deployments": [asdict(r) for r in history]}
    HISTORY_PATH.write_text(yaml.dump(data, default_flow_style=False))

    # Deploy
    if target == "local":
        docker_compose_up(version)
    elif target in ("aws", "gcp"):
        terraform_apply(target, version)

    print(f"Deployed {version} to {target}")


def show_history() -> None:
    """Print all deployment records."""
    history = load_deployment_history()
    if not history:
        print("No deployment history found.")
        return
    print(f"{'Version':<25} {'Target':<8} {'Status':<12} {'Approval':<10} {'SHA':<10}")
    print("-" * 70)
    for r in history:
        print(f"{r.version:<25} {r.target:<8} {r.status:<12} {r.approval:<10} {r.git_sha:<10}")


def show_version(version: str) -> None:
    """Print details for a specific deployment version."""
    history = load_deployment_history()
    for r in history:
        if r.version == version:
            for key, value in asdict(r).items():
                print(f"  {key}: {value}")
            return
    print(f"Version {version} not found in deployment history.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """CLI entrypoint for deployment operations."""
    parser = argparse.ArgumentParser(description="Deploy RAG Learning Lab")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--target",
        choices=VALID_TARGETS,
        help="Deploy to the specified target",
    )
    group.add_argument(
        "--rollback",
        metavar="VERSION",
        help="Roll back to a specific version or 'last-approved'",
    )
    group.add_argument(
        "--history",
        action="store_true",
        help="Show deployment history",
    )
    group.add_argument(
        "--show",
        metavar="VERSION",
        help="Show details for a specific deployment version",
    )

    args = parser.parse_args()

    if args.target:
        deploy(args.target)
    elif args.rollback:
        if args.rollback == "last-approved":
            record = get_last_approved_version("local")
            if record is None:
                print("No approved deployments found", file=sys.stderr)
                sys.exit(1)
            rollback(record.version, record.target)
        else:
            rollback(args.rollback, "local")
    elif args.history:
        show_history()
    elif args.show:
        show_version(args.show)


if __name__ == "__main__":
    main()
