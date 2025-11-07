import json
import carla
import argparse
import xml.etree.ElementTree as ET
from agents.navigation.global_route_planner import GlobalRoutePlanner
import os
import atexit
import subprocess
import time
import random
import numpy as np
import matplotlib.pyplot as plt


# v5 : Completion和Ability都只按优先级选取失败的一个原因

# Ability = {
#     "Overtaking":['Accident', 'AccidentTwoWays', 'ConstructionObstacle', 'ConstructionObstacleTwoWays', 'HazardAtSideLaneTwoWays', 'HazardAtSideLane', 'ParkedObstacleTwoWays', 'ParkedObstacle', 'VehicleOpenDoorTwoWays'],
#     "Merging": ['CrossingBicycleFlow', 'EnterActorFlow', 'HighwayExit', 'InterurbanActorFlow', 'HighwayCutIn', 'InterurbanAdvancedActorFlow','VehicleOpensDoorTwoWays', 'MergerIntoSlowTrafficV2', 'MergeIntoSlowTraffic', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow', 'ParkingExit', 'SequentialLaneChange', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow'],
#     "Emergency_Brake": ['BlockedIntersection', 'DynamicObjectCrossing', 'HardBreakRoute', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'ParkingCutIn', 'PedestrianCrossing', 'ParkingCrossingPedestrain', 'StaticCutIn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'ControlLoss'],
#     "Give_Way": ['InvadingTurn', 'YieldToEmergencyVehicle'],
#     "Traffic_Signs": ['BlockedIntersection', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'EnterActorFlow', 'CrossingBicycleFlow', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow', 'T_Junction', 'VanillaNonSignalizedTurn', 'VanillaSignalizedTurnEncounterGreenLight', 'VanillaSignalizedTurnEncounterRedLight', 'VanillaNonSignalizedTurnEncouterStopsign', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian']
# }

Ability = {
    "Overtaking":['Accident', 'AccidentTwoWays', 'ConstructionObstacle', 'ConstructionObstacleTwoWays', 'HazardAtSideLaneTwoWays', 'HazardAtSideLane', 'ParkedObstacleTwoWays', 'ParkedObstacle', 'VehicleOpensDoorTwoWays'],
    "Merging": ['CrossingBicycleFlow', 'EnterActorFlow', 'HighwayExit', 'InterurbanActorFlow', 'HighwayCutIn', 'InterurbanAdvancedActorFlow', 'MergerIntoSlowTrafficV2', 'MergerIntoSlowTraffic', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow', 'ParkingExit', 'SequentialLaneChange', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow'],
    "Emergency_Brake": ['BlockedIntersection', 'DynamicObjectCrossing', 'HardBreakRoute', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'ParkingCutIn', 'PedestrianCrossing', 'ParkingCrossingPedestrian', 'StaticCutIn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'ControlLoss'],
    "Give_Way": ['InvadingTurn', 'YieldToEmergencyVehicle'],
    "Traffic_Signs": ['BlockedIntersection', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'EnterActorFlow', 'CrossingBicycleFlow', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'NonSignalizedJunctionLeftTurnEnterFlow', 'OppositeVehicleTakingPriority', 'OppositeVehicleRunningRedLight', 'PedestrianCrossing', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'SignalizedJunctionLeftTurnEnterFlow', 'T_Junction', 'VanillaNonSignalizedTurn', 'VanillaSignalizedTurnEncounterGreenLight', 'VanillaSignalizedTurnEncounterRedLight', 'VanillaNonSignalizedTurnEncounterStopsign', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian']
}

# failed_dict = {
#     "Overtaking":"Overtaking_failed",
#     "Merging":"Merging_failed",
#     "Emergency_Brake":"Emergency_Brake_failed",
#     "Give_Way":"Give_Way_failed",
#     "Traffic_Signs":"Traffic_Signs_failed",
#     "Other":"Other_failed"
# }

"""
{  
    "能力": {  
        "超车": [  
            "事故", "双向事故", "施工障碍", "双向施工障碍", "双向辅道危险", "辅道危险", "双向停车障碍", 
            "停车障碍", "双向车辆开门"  
        ],  
        "汇入": [  
            "穿越自行车流", "进入车流", "高速出口", "城际车流", "高速切入", "城际优先车流", 
            "汇入慢行交通V2", "汇入慢行交通", "无信号路口左转", "无信号路口右转", "无信号路口左转进入车流",
            "停车场出口", "连续变道", "信号路口左转", "信号路口右转", "信号路口左转进入车流"  
        ],  
        "紧急制动": [  
            "路口阻塞", "动态物体穿越", "紧急制动路线", "对面车辆优先", "对面车辆闯红灯", "停车切入", 
            "行人过街", "停车穿越行人", "静态切入", "车辆转弯路线", "车辆转弯路线涉及行人", "失控"  
        ],  
        "让行": [  
            "转弯侵入", "为紧急车辆让行"  
        ],  
        "交通标志": [  
            "路口阻塞", "对面车辆优先", "对面车辆闯红灯", "行人过街", "车辆转弯路线", 
            "车辆转弯路线涉及行人", "进入车流", "穿越自行车流", "无信号路口左转", "无信号路口右转", 
            "无信号路口左转进入车流", "对面车辆优先", "对面车辆闯红灯", "行人过街", "信号路口左转", 
            "信号路口右转", "信号路口左转进入车流", "T型路口", "普通无信号转弯", "普通信号转弯遇绿灯",
            "普通信号转弯遇红灯", "普通无信号转弯遇停车标志", "车辆转弯路线", "车辆转弯路线涉及行人"  
        ]  
    }  
}

"""
def get_infraction_status(record):
    for infraction,  value in record['infractions'].items():
        if infraction == "min_speed_infractions":
            continue
        elif len(value) > 0:
            return True
    return False

def update_Ability(scenario_name, Ability_Statistic, status):
    for ability, scenarios in Ability.items():
        if scenario_name in scenarios:
            Ability_Statistic[ability][1] += 1
            if status:
                Ability_Statistic[ability][0] += 1
    pass

def update_Success(scenario_name, Success_Statistic, status):
    if scenario_name not in Success_Statistic:
        if status:
            Success_Statistic[scenario_name] = [1, 1]
        else:
            Success_Statistic[scenario_name] = [0, 1]
    else:
        Success_Statistic[scenario_name][1] += 1
        if status:
            Success_Statistic[scenario_name][0] += 1
    pass

def get_position(xml_route):
    waypoints_elem = xml_route.find('waypoints')
    keypoints = waypoints_elem.findall('position')
    return [carla.Location(float(pos.get('x')), float(pos.get('y')), float(pos.get('z'))) for pos in keypoints]

def get_route_result(records, route_id):
    for record in records:
        record_route_id = record['route_id'].split('_')[1]
        if route_id == record_route_id:
            return record
    return None

def get_waypoint_route(locs, grp):
    route = []
    for i in range(len(locs) - 1):
        loc = locs[i]
        loc_next = locs[i + 1]
        interpolated_trace = grp.trace_route(loc, loc_next)
        for wp, _ in interpolated_trace:
            route.append(wp)
    return route

def main(args):
    routes_file = args.file 
    result_file = args.result_file
    Ability_Statistic = {}
    crash_route_list = []
    for key in Ability:
        Ability_Statistic[key] = [0, 0.]
    Success_Statistic = {}
    
    with open(result_file, 'r') as f:
        json_file = json.load(f)
    records = json_file["_checkpoint"]["records"]
                    
    tree = ET.parse(routes_file)
    root = tree.getroot()
    routes = root.findall('route')
    sorted_routes = sorted(routes, key=lambda x: x.get('town'))
    
    
    carla_path = os.environ["CARLA_ROOT"]
    cmd1 = f"{os.path.join(carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={args.port}"
    server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
    print(cmd1, server.returncode, flush=True)
    time.sleep(10)
    client = carla.Client(args.host, args.port)
    client.set_timeout(300)
    
    current_town = sorted_routes[0].get('town')
    world = client.load_world(current_town)
    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, 1.0)
    
    status_col = ['no_infruction', 'Completed', 'not_Completed', 'failed', 'success', 'success_junction', 'fail_junction', 'collision', 'red_light', 'block', 'timeout', 'other', 'stop', 'yield']
    results = {ability: [] for ability in Ability.keys()}
    results.update({'all_route':[], 'completed':[], 'completed_failed':[],'not_completed':[]})

    for route in sorted_routes:
        print("___________")
        scenarios = route.find('scenarios')
        scenario_name = scenarios.find('scenario').get("type")
        route_id = route.get('id')
        route_record = get_route_result(records, route_id)
        
        
        ###########
        infractions = route_record['infractions']
        score_composed = route_record['scores']['score_composed']
        
        # dict = {
        #     "collision": False,
        #     "red_light":False,
        #     "block":False,
        #     "stop":False,
        #     "yield":False,
        #     "other":False,
        #     "timeout":False
        # }
        
        # Determine status category
        if route_record["status"] == "Completed":
            results['all_route'].append({
                        'scenario_name': scenario_name,
                        'status': "Completed"
                    })
            if score_composed == 100.:
                status_category = 'success'
                results['completed'].append({
                        'scenario_name': scenario_name,
                        'status': 'success'
                    })
            else:
                results['completed'].append({
                        'scenario_name': scenario_name,
                        'status': 'failed'
                    })

                for infra_key, infra_value in infractions.items():
                    if 'collision' in infra_key and infra_value != []:
                        complete_category = 'collision'
                        break
                    elif 'red_light' in infra_key and infra_value != []:
                        complete_category = 'red_light'
                        break
                    elif 'blocked' in infra_key and infra_value != []:
                        complete_category = 'block'
                        break
                    elif 'stop' in infra_key and infra_value != []:
                        complete_category = 'stop'
                        break
                    elif 'yield' in infra_key and infra_value != []:
                        complete_category = 'yield'
                        break
                    elif infra_value != [] and 'speed' in infra_key:
                        pass
                    elif infra_value != [] and 'speed'  not in infra_key:
                        complete_category = 'other'
                        break
                    else:
                        pass
                
                
                results['completed_failed'].append({
                        'scenario_name': scenario_name,
                        'status': complete_category})
                    
                # for k, v in dict.items():
                #     if v is True:
                #         results['completed_failed'].append({
                #             'scenario_name': scenario_name,
                #             'status': k})
                        
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
                
                
        else:
            # not completed
            results['all_route'].append({
                        'scenario_name': scenario_name,
                        'status': "not_Completed"
                    })
            complete_category = ""
            for infra_key, infra_value in infractions.items():
                if 'collision' in infra_key and infra_value != []:
                    complete_category = 'collision'
                    break
                elif 'red_light' in infra_key and infra_value != []:
                    complete_category = 'red_light'
                    break
                elif 'blocked' in infra_key and infra_value != []:
                    complete_category = 'block'
                    break
                elif 'stop' in infra_key and infra_value != []:
                    complete_category = 'stop'
                    break
                elif 'yield' in infra_key and infra_value != []:
                    complete_category = 'yield'
                    break
                elif infra_value != [] and 'speed' in infra_key:
                    pass
                elif infra_value != [] and 'speed'  not in infra_key:
                    complete_category = 'other'
                    break
                else:
                    pass
            if complete_category is "":
                complete_category = 'no_infruction'
            
                
            results['not_completed'].append({
                    'scenario_name': scenario_name,
                    'status': complete_category})
                    
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
        
        
        ###########
        
        
        # print(scenarios)
        # exit()
        
        if route_record is None:
            crash_route_list.append((scenario_name, route_id))
            print('No result record of route', route_id, "in the result file")
            continue
        if route_record["status"] == 'Completed' or route_record["status"] == "Perfect":
            if get_infraction_status(route_record):
                record_success_status = False
            else:
                record_success_status = True
        else:
            record_success_status = False
        update_Ability(scenario_name, Ability_Statistic, record_success_status)
        update_Success(scenario_name, Success_Statistic, record_success_status)
        # if scenario_name in Ability["Traffic_Signs"] and (scenario_name in Ability["Merging"] or scenario_name in Ability["Emergency_Brake"]):
        # Only these three 'Ability's intersect
        if scenario_name in Ability["Traffic_Signs"]:
            # Only these three 'Ability's intersect
            # print(route.get('town'))
            if route.get('town') != current_town:
                current_town = route.get('town')
                print("Loading the town:", current_town)
                world = client.load_world(current_town)
                print("successfully load the town:", current_town)
            carla_map = world.get_map()
            grp = GlobalRoutePlanner(carla_map, 1.0)
            
            # waypoints position(x,y,z)
            location_list = get_position(route)
            # print(len(location_list))
            # print(location_list)
            waypoint_route = get_waypoint_route(location_list, grp)
            # print(waypoint_route)
            # print(len(waypoint_route))
            # exit()
            count = 0
            for wp in waypoint_route:
                count += 1
                if wp.is_junction:
                    break 
            if not wp.is_junction:
                raise RuntimeError("This route does not contain any junction-waypoint!")
            # +8 to ensure the ego pass the trigger volume
            junction_completion = float(count+8) / float(len(waypoint_route))
            record_completion = route_record["scores"]["score_route"] / 100.0
            stop_infraction = route_record["infractions"]["stop_infraction"]
            red_light_infraction = route_record["infractions"]["red_light"]
            if record_completion > junction_completion and not stop_infraction and not red_light_infraction:
                Ability_Statistic['Traffic_Signs'][0] += 1
                Ability_Statistic['Traffic_Signs'][1] += 1
                results["Traffic_Signs"].append({
                'scenario_name': scenario_name,
                'status': "success_junction"
                })
            else:
                Ability_Statistic['Traffic_Signs'][1] += 1
                results["Traffic_Signs"].append({
                'scenario_name': scenario_name,
                'status': "fail_junction"
                })
        else:
            pass
        
    Ability_Res = {}
    for ability, statis in Ability_Statistic.items():
        Ability_Res[ability] = float(statis[0])/float(statis[1])
        print("ability:", ability)
        print("statis[0]:", statis[0])
        print("statis[1]:", statis[1])
        
    for key, value in Ability_Res.items():
        print(key, ": ", value)
    
    Ability_Res['mean'] = sum(list(Ability_Res.values())) / 5
    Ability_Res['crashed'] = crash_route_list
    with open(f"{result_file.split('.')[0]}_ability.json", 'w') as file:
        json.dump(Ability_Res, file, indent=4)
        
    Success_Res = {}
    Route_num = 0
    Succ_Route_num = 0
    for scenario, statis in Success_Statistic.items():
        Success_Res[scenario] = float(statis[0])/float(statis[1])
        Succ_Route_num += statis[0]
        Route_num += statis[1]
    assert len(crash_route_list) == 220 - float(Route_num)
    print(f'Crashed Route num: {len(crash_route_list)}, Crashed Route ID: {crash_route_list}')
    print('Finished!')
    
    # graph
    
    # # Generate pie charts
    # # Use consistent colors for categories
    colors = {'no_infruction':'green', 'not_Completed':'red', 'Completed':'green','failed':'red', 'success_junction': 'orange', 'fail_junction': 'cyan', 'success': 'green', 'collision': 'red', 'timeout': 'orange', 'red_light': 'blue', 'block': 'purple', 'other': 'gray', 'stop': 'cyan', 'yield': 'yellow'}


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
    
    
    status_counts = {status: 0 for status in status_col}
    for record in results['completed']:
        status_counts[record['status']] += 1
    count_dict = {status: count for status, count in status_counts.items() if count > 0}
    print("completed:")
    print(count_dict)
    
    status_counts = {status: 0 for status in status_col}
    for record in results['completed_failed']:
        status_counts[record['status']] += 1
    count_dict = {status: count for status, count in status_counts.items() if count > 0}
    print("completed_failed:")
    print(count_dict)
    
    status_counts = {status: 0 for status in status_col}
    for record in results['not_completed']:
        status_counts[record['status']] += 1
    count_dict = {status: count for status, count in status_counts.items() if count > 0}
    print("not_completed:")
    print(count_dict)
        

    # print("Traffic_Signs:", ability_stats["Traffic_Signs"])
    # exit()

    # Create a single figure with subplots
    num_abilities = len(ability_stats)
    # print(num_abilities)
    # exit()
    grid_size = int(np.ceil(np.sqrt(num_abilities)))
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(5*grid_size, 5*grid_size))
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
    
    # print(set([item['scenario_name'] for item in results['Other']]))

if __name__=='__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-f', '--file', nargs=None, default="leaderboard/data/bench2drive220.xml", help='route file')
    argparser.add_argument('-r', '--result_file', nargs=None, default="", help='result json file')
    argparser.add_argument('-t', '--host', default='localhost', help='IP of the host server (default: localhost)')
    argparser.add_argument('-p', '--port', nargs=1, default=2000, help='carla rpc port')
    args = argparser.parse_args()
    main(args)
    