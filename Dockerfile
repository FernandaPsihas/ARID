# ARID pipeline runner. qdrant + ollama run as their own containers (see compose).
FROM python:3.12-slim

# build-essential: tree-sitter wheels compile C extensions. git: extract.py uses `git ls-files`.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global --add safe.directory '*'

WORKDIR /app

# deps first so they cache across code edits. requirements.txt installs the vendored
# tree-sitter-jsonnet from its local path (patched setup.py adds the src/scanner.c the
# published wheel omits), so this one command builds everything — same as on Windows.
COPY requirements.txt ./
COPY tree-sitter-jsonnet ./tree-sitter-jsonnet
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# compose mounts the repo over /app at runtime; this COPY just makes the image runnable standalone.
CMD ["bash"]
