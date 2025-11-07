export CARLA_ROOT=/home/user/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner
    
# Merge eval json and get driving score and success rate
# This script will assume the total number of routes with results is 220. If there is not enough, the missed ones will be treated as 0 score.
python tools/merge_route_json.py -f your_folder

# Get multi-ability results
python tools/ability_benchmark5.py -r /home/user/new_Bench2Drive/Bench2Drive/diffad-v1.3-8epoch-drop1-deviate-0.3_b2d_traj/merged.json

# Get driving efficiency and driving smoothness results
# python tools/efficiency_smoothness_benchmark.py -f /home/user/new_Bench2Drive/Bench2Drive/diffad-fp-0.5-plan-3s-dpm-10_b2d_traj/merged.json -m /home/user/new_Bench2Drive/Bench2Drive/eval_bench2drive220_1_diffad-fp-0.5-plan-3s-dpm-10_traj/
# python tools/efficiency_smoothness_benchmark.py -f /home/user/new_Bench2Drive/Bench2Drive/vad_b2d_traj_2/merged.json -m /home/user/new_Bench2Drive/Bench2Drive/vad_b2d_traj_2