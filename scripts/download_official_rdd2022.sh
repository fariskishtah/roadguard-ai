#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/raw}"
archive="$destination/RDD2022_released_through_CRDDC2022.zip"
expected_md5="b62bd51d2ffcfaa76c60f234f0cc2bb3"
required_kib=41943040

mkdir -p "$destination"
available_kib="$(df -Pk "$destination" | awk 'NR==2 {print $4}')"
if [ "$available_kib" -lt "$required_kib" ]; then
  echo "RDD2022 extraction requires at least 40 GiB free; only $((available_kib / 1048576)) GiB is available." >&2
  exit 2
fi

curl \
  --fail \
  --location \
  --continue-at - \
  --output "$archive" \
  "https://ndownloader.figshare.com/files/38030910"

actual_md5="$(md5 -q "$archive")"
if [ "$actual_md5" != "$expected_md5" ]; then
  echo "Checksum mismatch: expected $expected_md5, received $actual_md5" >&2
  exit 3
fi

unzip -n -q "$archive" -d "$destination/extracted"

