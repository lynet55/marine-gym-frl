#!/usr/bin/env bash
# Checks that the MarineGym submodule is checked out, then runs the given command.

if [ ! -f /work/marinegym/setup.py ]; then
    echo "MarineGym submodule is missing. On the host, run: git submodule update --init" >&2
    exit 1
fi

exec "$@"
