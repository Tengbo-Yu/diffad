import json
import numpy as np
import matplotlib.pyplot as plt

# Define the abilities dictionary
Ability = {
    "Overtaking":['Accident', 'AccidentTwoWays', 'ConstructionObstacle', 'ConstructionObstacleTwoWays', 'HazardAtSideLaneTwoWays', 'HazardAtSideLane', 'ParkedObstacleTwoWays', 'ParkedObstacle', 'VehicleOpenDoorTwoWays'],
    "Merging": ['CrossingBicycleFlow', 'EnterActorFlow', 'HighwayExit', 'InterurbanActorFlow', 'HighwayCutIn', 'InterurbanAdvancedActorFlow', 'MergerIntoSlowTrafficV2', 'MergeIntoSlowTraffic', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow', 'ParkingExit', 'SequentialLaneChange', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow'],
    "Emergency_Brake": ['BlockedIntersection', 'DynamicObjectCrossing', 'HardBreakRoute', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'ParkingCutIn', 'PedestrianCrossing', 'ParkingCrossingPedestrain', 'StaticCutIn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'ControlLoss'],
    "Give_Way": ['InvadingTurn', 'YieldToEmergencyVehicle'],
    "Traffic_Signs": ['BlockedIntersection', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'EnterActorFlow', 'CrossingBicycleFlow', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow', 'T_Junction', 'VanillaNonSignalizedTurn', 'VanillaSignalizedTurnEncounterGreenLight', 'VanillaSignalizedTurnEncounterRedLight', 'VanillaNonSignalizedTurnEncouterStopsign', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian']
}

# Load JSON file
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {file_path}")
        return None

json_path = "/home/user/new_Bench2Drive/Bench2Drive/diffad-fp-0.5-plan-3s-dpm-10_b2d_traj/merged.json"
json_file = read_json_file(json_path)

records = json_file['_checkpoint']['records']

# Status categories
status_col = ['success', 'collision', 'red_light', 'block', 'timeout', 'other', 'stop', 'yield']
results = {ability: [] for ability in Ability.keys()}
results.update({'Other':[]})

# print(results)
# exit()

# suc_count = 0
# Classify records into abilities and statuses
for record in records:
    scenario_name = '_'.join(record['scenario_name'].split('_')[:-1])
    # print(scenario_name)
    # exit()
    infractions = record['infractions']
    score_composed = record['scores']['score_composed']

    # Determine status category
    if score_composed == 100.:
        status_category = 'success'
        # suc_count += 1
    else:
        for infra_key, infra_value in infractions.items():
            if 'collision' in infra_key and infra_value != []:
                status_category = 'collision'
                break
            elif 'red_light' in infra_key and infra_value != []:
                status_category = 'red_light'
                break
            elif 'blocked' in infra_key and infra_value != []:
                status_category = 'block'
                break
            elif 'stop' in infra_key and infra_value != []:
                status_category = 'stop'
                break
            elif 'yield' in infra_key and infra_value != []:
                status_category = 'yield'
                break
            elif infra_value != [] and 'speed' not in infra_key:
                status_category = 'other'
                break
            else:
                status_category = 'timeout'

    # Match route name with ability and assign status
    found_ability = False
    for ability, value in Ability.items():
        if scenario_name in value:
            results[ability].append({
                'scenario_name': scenario_name,
                'status': status_category
            })
            found_ability = True
            # break

    if not found_ability:
        results['Other'].append({
            'scenario_name': scenario_name,
            'status': status_category
        })

# print(results)
# exit()


# # Generate pie charts
# # Use consistent colors for categories
colors = {'success': 'green', 'collision': 'red', 'timeout': 'orange', 'red_light': 'blue', 'block': 'purple', 'other': 'gray', 'stop': 'cyan', 'yield': 'yellow'}


# overall stats
status_counts = {status: 0 for status in status_col}
for ability, data in results.items():
    for record in data:
        status_counts[record['status']] += 1
total = sum(status_counts.values())
proportions = {status: count / total for status, count in status_counts.items() if count > 0}

print('proportions=', proportions, '\n', 'status_counts=', status_counts)

# # # Calculate status proportions and counts for each ability
ability_stats = {}
for ability, data in results.items():
    status_counts = {status: 0 for status in status_col}
    for record in data:
        status_counts[record['status']] += 1
    total = sum(status_counts.values())
    proportions = {status: count / total for status, count in status_counts.items() if count > 0}
    ability_stats[ability] = {'proportions': proportions, 'counts': status_counts}

# print("Traffic_Signs:", ability_stats["Traffic_Signs"])
# exit()

# Create a single figure with subplots
num_abilities = len(ability_stats)
grid_size = int(np.ceil(np.sqrt(num_abilities)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(5*grid_size, 5*grid_size))
# fig.suptitle('Status Distribution for All Abilities', fontsize=16, fontweight='bold')

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot each ability's pie chart
for idx, (ability, stats) in enumerate(ability_stats.items()):
    ax = axes[idx]
    labels = list(stats['proportions'].keys())
    sizes = list(stats['proportions'].values())
    counts = [stats['counts'][label] for label in labels]

    def autopct_format(pct, allvalues):
        absolute = int(round(pct/100.*sum(allvalues)))
        return f"{pct:.1f}%\n({absolute:d})"

    # Use consistent colors for each status
    ax.pie(sizes, labels=labels, autopct=lambda pct: autopct_format(pct, counts), colors=[colors[label] for label in labels], startangle=90)
    ax.set_title(f'{ability}', fontsize=12, fontweight='bold')

# Remove any unused subplots
for idx in range(num_abilities, len(axes)):
    fig.delaxes(axes[idx])

# Add a legend for clarity
fig.legend([plt.Line2D([0], [0], color=colors[key], lw=4) for key in colors.keys()], colors.keys(), loc='upper right')

plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05)
plt.savefig('all_abilities_status_distribution.png', bbox_inches='tight')
plt.close()



print(set([item['scenario_name'] for item in results['Other']]))