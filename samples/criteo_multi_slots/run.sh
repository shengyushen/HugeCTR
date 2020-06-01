#../../bin/huge_ctr  --train ./criteo.json
nvprof --csv --print-api-trace  --print-gpu-trace --print-nvlink-topology --print-pci-topology \
  -o "criteo_%h_%p.prof" \
  ../../bin/huge_ctr  --train ./criteo.json 2> criteo.log
