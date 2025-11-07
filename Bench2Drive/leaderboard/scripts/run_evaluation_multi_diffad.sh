#!/bin/bash
# bash leaderboard/scripts/run_evaluation_multi_diffad.sh 
# bash tools/clean_carla.sh
export DiffAD_ROOT=/home/users/tao06.wang/server_code/diffad
export PYTHONPATH=$PYTHONPATH:${DiffAD_ROOT}
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220


TEAM_AGENT=leaderboard/team_code_zoo/diffad_b2d_agent.py
# Must set YOUR_CKPT_PATH
TEAM_CONFIG=Bench2DriveZoo/diffad/configs/config_b2d_carla.yaml+path/to/model.pt

BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=diffad-E2E
SAVE_PATH=./eval_bench2drive220_1_${ALGO}_${PLANNER_TYPE}

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi



echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
# Example, 8*H100, 1 task per gpu
# GPU_RANK_LIST=(0 1 2 3 4 5 6 7)
# TASK_LIST=(0 1 2 3 4 5 6 7)
TASK_NUM=2
GPU_RANK_LIST=(0 1)
TASK_LIST=(0 1)
echo -e "\033[32m GPU_RANK_LIST: $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST: $TASK_LIST \033[0m"
echo -e "***********************************************************************************"

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    echo -e "\033[33m TASK_NUM:$TASK_NUM \033[0m"
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi



length=${#GPU_RANK_LIST[@]}


start_task() {
    local i=$1
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    
    echo -e "\033[32m Starting task $i \033[0m"
    
    # Kill Python and Carla processes for this task
    # lsof -i:$PORT -t | xargs -I {} bash -c 'ps -p {} -o comm= | grep -E "python|CarlaUE4" && kill -9 {}'
    pkill -f "python.*${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py.*--port=${PORT}"
    echo -e " bash kill -9 $(lsof -i:$PORT -t)"

    # Wait a moment to ensure processes are terminated
    sleep 2

    # Start the task
    bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log &
    echo -e "bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK"
}

for ((i=0; i<$length; i++ )); do
    start_task $i
    sleep 5
done


check_and_restart_task() {
    local i=$1
    local log_file=$2
    local timeout=300  # 5 minutes, adjust as needed

    if [ -f "$log_file" ]; then
        # Check for engine crash
        if grep -q "Engine crash handling finished" "$log_file"; then
            echo -e "\033[31m Task $i has crashed (Engine crash found). Restarting... \033[0m"
            start_task $i
            sleep 5
            return
        fi

        # Check for timeout
        last_modified=$(stat -c %Y "$log_file")
        current_time=$(date +%s)
        time_diff=$((current_time - last_modified))

        if [ $time_diff -gt $timeout ]; then
            echo -e "\033[33m Task $i appears to be stuck (no log updates for ${timeout}s). Restarting... \033[0m"
            start_task $i
            sleep 5
        fi
    fi
}
# Monitor and restart loop
while true; do
    for ((i=0; i<$length; i++ )); do
        log_file="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log"
        check_and_restart_task $i $log_file
    done
    sleep 60  # Check every minute
done

wait
