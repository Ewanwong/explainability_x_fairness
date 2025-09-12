#!/usr/bin/env bash

bash decoder_scripts/compute_fairness_race_test.sh
bash decoder_scripts/compute_fairness_religion_test.sh
bash decoder_scripts/compute_fairness_gender_test.sh

# bash decoder_scripts/compute_fairness_race_val.sh
# bash decoder_scripts/compute_fairness_religion_val.sh
# bash decoder_scripts/compute_fairness_gender_val.sh