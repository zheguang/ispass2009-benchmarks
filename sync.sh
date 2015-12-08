#!/bin/bash

git diff --cached > /tmp/sync.diff

host=sam@bsn04.cs.brown.edu
repo=/data/sam/studio/research/gpgpu

scp /tmp/sync.diff "$host":"$repo"

ssh $host "cd $repo && git reset --hard HEAD && git apply sync.diff"

