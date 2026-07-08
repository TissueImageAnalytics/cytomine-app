#!/usr/bin/env bash
# Build the App Engine bundle (image tar + descriptor) as a .zip ready to upload.
# Keep VERSION in sync with `version` and `configuration.image.file` in descriptor.yml.
set -euo pipefail

VERSION="0.1.6"
NAMESPACE="com.cytomine.nuclei.segmentation.interactive.nuclick"

IMAGE="${NAMESPACE//.//}:${VERSION}"   # slash form, e.g. com/cytomine/.../nuclick:0.1.0
ARTIFACT="${NAMESPACE}-${VERSION}"

cd "$(dirname "$0")"

sudo docker build --provenance=false --sbom=false -t "${IMAGE}" .
sudo docker save "${IMAGE}" -o "${ARTIFACT}.tar"
sudo chown "${USER}:${USER}" "${ARTIFACT}.tar"

rm -f "${ARTIFACT}.zip"
zip -0 "${ARTIFACT}.zip" descriptor.yml "${ARTIFACT}.tar"

echo "Built ${ARTIFACT}.zip"
