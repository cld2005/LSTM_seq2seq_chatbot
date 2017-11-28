#!/usr/bin/env bash
scp -i ../transfer.pem setup.sh ubuntu@$1:~/
