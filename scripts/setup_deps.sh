#!/usr/bin/env bash
# Create a Python 3.12 venv (sandbox parity), install nova_miner + deps, download DB.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOVA_MINER_SRC="${NOVA_MINER_SRC:-/root/nova-blueprint/libs/nova_miner}"
VENV="${ROOT}/.venv"
DB_URL="${MOL_RXN_DB_URL:-https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/sanqi/molecules.sqlite}"

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "==> ensuring Python 3.12 + venv at ${VENV}"
uv python install 3.12
uv venv --python 3.12 "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "==> installing sandbox pip deps"
uv pip install -r "${NOVA_MINER_SRC}/requirements/sandbox.txt" numpy

DEST="$(python -c 'import site; print(site.getsitepackages()[0])')/nova_miner"
echo "==> installing nova_miner into ${DEST}"
mkdir -p "${DEST}"
rsync -a --delete \
  --exclude 'combinatorial_db/molecules.sqlite' \
  --exclude 'combinatorial_db/__pycache__' \
  --exclude '__pycache__' \
  "${NOVA_MINER_SRC}/" "${DEST}/"

mkdir -p "${DEST}/combinatorial_db"
DB_PATH="${DEST}/combinatorial_db/molecules.sqlite"
if [[ ! -f "${DB_PATH}" ]]; then
  echo "==> downloading molecules.sqlite"
  curl -L --fail --retry 3 -o "${DB_PATH}.partial" "${DB_URL}"
  mv "${DB_PATH}.partial" "${DB_PATH}"
else
  echo "==> molecules.sqlite already present ($(du -h "${DB_PATH}" | cut -f1))"
fi

echo "==> building molecule_roles index (idempotent)"
python "${DEST}/combinatorial_db/build_index.py" "${DB_PATH}"

python - <<'PY'
import nova_miner
from pathlib import Path
db = Path(nova_miner.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite"
assert db.is_file(), db
print(f"nova_miner OK at {nova_miner.__file__}")
print(f"db OK ({db.stat().st_size:,} bytes) at {db}")
PY

cat > "${ROOT}/.python-path" <<EOF
${VENV}/bin/python
EOF

echo "==> deps ready"
echo "Use: source ${VENV}/bin/activate"
echo "Or:  ${VENV}/bin/python …"
