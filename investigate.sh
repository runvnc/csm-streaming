#!/bin/bash
echo "=== DISK SPACE OVERVIEW ==="
df -h | grep -v 'thermal_throttle\|/sys/\|/proc/' | head -30

echo ""
echo "=== INODE USAGE ==="
df -i | grep -v 'thermal_throttle\|/sys/\|/proc/' | head -20

echo ""
echo "=== /tmp DETAILS ==="
df -h /tmp
mount | grep -E '^tmpfs.*/tmp'

echo ""
echo "=== TMPDIR ENV ==="
echo "TMPDIR=$TMPDIR"
echo "TEMP=$TEMP"
echo "TMP=$TMP"

echo ""
echo "=== /tmp CONTENTS ==="
du -sh /tmp/* 2>/dev/null | sort -h | tail -20

echo ""
echo "=== /tmp FILESYSTEM TYPE ==="
stat -f /tmp

echo ""
echo "=== CHECKING IF /tmp IS TMPFS (RAM-BASED) ==="
findmnt /tmp

echo ""
echo "=== PYTHON'S TEMPFILE DEFAULT ==="
python3 -c "import tempfile; print('tempfile.gettempdir():', tempfile.gettempdir())"

echo ""
echo "=== PIP CONFIG ==="
pip config list 2>/dev/null || echo "No pip config"

echo ""
echo "=== CHECKING BUILD ISOLATION TEMP DIR ==="
echo "pip uses its own --prefix for build isolation, ignoring TMPDIR"
echo "The --prefix /tmp/pip-build-env-* is hardcoded behavior"

echo ""

