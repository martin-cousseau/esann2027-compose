#!/bin/bash
# Follow-up jobs for a runtime whose first chain has ended (2026-11-15, DECISIONS.md #58).
# The orchestrating session places /content/out/NEXT with one line "<commit> <chain script>"; this loop checks that
# commit of the branch out into its own folder and runs colab/<chain script> from there, then waits for the next
# marker ("<any> stop" ends the loop). It lets a job whose code was not written when the runtime started run without
# another start command. Result files of the layer-level families are copied to the Hub every 10 minutes meanwhile.
export ESANN_OUT=/content/out
mkdir -p /content/out/logs/runner
( while true; do python /content/work/scripts/hub_sync.py put results/v2/E2 results/v2/E9 >/dev/null 2>&1; sleep 600; done ) &
SYNC=$!
while true; do
  while [ ! -f /content/out/NEXT ]; do sleep 30; done
  read -r COMMIT CHAIN < /content/out/NEXT
  mv /content/out/NEXT "/content/out/NEXT.taken.$(date +%s)"
  [ "$CHAIN" = "stop" ] && break
  case "$CHAIN" in chain_*.sh) ;; *) echo "next_loop: refused $CHAIN"; continue;; esac
  D="/content/work_${CHAIN%.sh}"
  rm -rf "$D"
  git clone -q -b claude/busy-hopper-ghtynk https://github.com/martin-cousseau/esann2027-compose "$D" \
    && (cd "$D" && git checkout -q "$COMMIT" && bash "colab/$CHAIN" > "/content/out/logs/runner/${CHAIN%.sh}.log" 2>&1)
  echo "next_loop: $CHAIN at $COMMIT ended with $?"
done
kill $SYNC 2>/dev/null
echo "NEXT LOOP DONE"
