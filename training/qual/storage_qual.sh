#!/bin/bash
# Storage semantics and throughput on the container disk (/root) and the global volume (/workspace).
set -u
T=$(date -u +%Y%m%dT%H%M%SZ)
for D in /root/stq /workspace/qual/stq-$T; do
  mkdir -p $D; echo "== $D"
  dd if=/dev/urandom of=/root/rand1g bs=1M count=1024 status=none 2>/dev/null || true
  [ -f /root/rand1g ] || exit 1
  S=$(date +%s.%N); cp /root/rand1g $D/f1g; sync -f $D/f1g; E=$(date +%s.%N)
  echo "write_1GiB_s $(python3 -c "print(round($E-$S,3))")"
  S=$(date +%s.%N); sha256sum $D/f1g | cut -c1-16; E=$(date +%s.%N); echo "read_hash_1GiB_s $(python3 -c "print(round($E-$S,3))")"
  sha256sum /root/rand1g | cut -c1-16
  # rename of a file onto an existing file: is the old content ever observable / does it succeed
  echo old > $D/target; echo new > $D/src; mv $D/src $D/target && echo "rename_rc=0 content=$(cat $D/target)"
  # directory rename
  mkdir -p $D/dirA && echo x > $D/dirA/a && mv $D/dirA $D/dirB && echo "dir_rename_ok=$(cat $D/dirB/a)"
  # many small files
  S=$(date +%s.%N); for i in $(seq 1 200); do echo $i > $D/small_$i; done; sync; E=$(date +%s.%N); echo "200_small_writes_s $(python3 -c "print(round($E-$S,3))")"
  python3 -c "import os;fd=os.open('$D/fs',os.O_CREAT|os.O_WRONLY);os.write(fd,b'x'*4096);os.fsync(fd);os.close(fd);print('fsync_ok')"
  chmod 600 $D/target; stat -c 'perm_after_chmod600=%a' $D/target
  python3 -c "import fcntl;f=open('$D/target');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('flock_ok')" 2>&1 | tail -1
  df -h $D | tail -1
done
rm -rf /root/stq
