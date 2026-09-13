#!/bin/bash

# Function to search for a value in all source files
find_in_sources() {
  local value="$1"
  local desc="$2"
  
  # Search in all source files
  if grep -qr "$value" FINDINGS.md GATES.md results/BENCHMARK_LANDSCAPE.md results/HEADLINE.md results/server_run1/M0_BASELINE_RESULTS.md 2>/dev/null; then
    echo "FOUND: $value ($desc)"
  else
    echo "NOT_FOUND: $value ($desc)"
  fi
}

# Quality curve values
find_in_sources "49\.4%" "IQ2_XXS score"
find_in_sources "62\.2%" "IQ2_S score"
find_in_sources "72\.0%" "Q2_K_XL score"
find_in_sources "82\.3%" "IQ3_XXS score"
find_in_sources "6\.76" "IQ2_XXS size"
find_in_sources "7\.79" "IQ2_S size"
find_in_sources "9\.14" "Q2_K_XL size"
find_in_sources "10\.17" "IQ3_XXS size"
find_in_sources "2\.12" "IQ2_XXS bpw"
find_in_sources "2\.45" "IQ2_S bpw"
find_in_sources "2\.87" "Q2_K_XL bpw"
find_in_sources "3\.20" "IQ3_XXS bpw"
find_in_sources "81/164" "IQ2_XXS tasks"
find_in_sources "102/164" "IQ2_S tasks"
find_in_sources "118/164" "Q2_K_XL tasks"
find_in_sources "135/164" "IQ3_XXS tasks"
find_in_sources "9\.7" "points per GiB"
find_in_sources "\+68%" "MTP speedup"

# Ablation values
find_in_sources "21\.09" "Q2_K_XL baseline"
find_in_sources "35\.11" "MTP n=2"
find_in_sources "35\.39" "MTP n=3"
find_in_sources "33\.63" "MTP n=4"
find_in_sources "86\.12" "ngram-mod"
find_in_sources "134\.40" "IQ3_XXS ngram"
find_in_sources "10212" "MiB baseline Q2"
find_in_sources "10588" "MiB MTP n=2 Q2"
find_in_sources "33\.83" "decode tok/s IQ3"

# Context scaling
find_in_sources "368 MiB" "per 16K"
find_in_sources "23 KiB/token" "KV size"
find_in_sources "21\.05" "ctx scaling"
find_in_sources "21\.00" "ctx flat"

# Bandwidth
find_in_sources "213\.5 GB/s" "effective bandwidth"
find_in_sources "59\.3%" "peak percentage"

# Best-of-n
find_in_sources "0\.678" "pass@1 51 tasks"
find_in_sources "0\.882" "pass@5 51 tasks"
find_in_sources "0\.600" "pass@1 164 tasks"
find_in_sources "0\.750" "oracle ceiling"
find_in_sources "0\.671" "verified-selection"
find_in_sources "0\.823" "temp-0 baseline"
find_in_sources "20\.4" "gate headroom"
find_in_sources "29" "temp-0 failures"

